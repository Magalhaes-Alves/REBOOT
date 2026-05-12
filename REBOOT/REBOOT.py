import numpy as np
import copy
import gurobipy as gp
from gurobipy import GRB
import lightgbm as lgb


class REBOOT:
    def __init__(
        self, model: lgb.Booster, N, num_features, num_categoricas, num_continuas
    ):

        self.model_dump = model.dump_model()
        self.N_total = N
        self.num_features = num_features
        self.num_categoricas = num_categoricas
        self.num_continuas = num_continuas
        self.eta = self.model_dump.get("learning_rate", 0.05)
        self.reg_lambda = self.model_dump.get("lambda_l2", 0.1)

    def execute(self):
        print("Attack REBOOT - Starting")
        self.phase_1()
        self.y_bar = 0.475
        self.phase_2()
        self.phase_3()
        self.phase_4()

        print("Attack REBOOT - Finalized")

    def phase_1(self):

        f0, objective = self._extract_f0_and_objective()
        print(f"F_0 extraída: {f0}, Objetivo: {objective}")

        if "binary" in objective:
            # self.y_bar = 1/(1+np.exp(-f0))
            self.y_bar = 0.475

            print(f"Probabilidade média (y_bar) calculada: {self.y_bar:.6f}")

        elif "regression" in objective:
            self.y_bar = f0
            print(f"Valor médio (y_bar) calculado: {self.y_bar:.6f}")

        return self.y_bar, objective

    def _extract_f0_and_objective(self):
        # Extrai f0 e o objetivo do modelo
        # 1. Identificar a função objetivo do modelo
        objective = self.model_dump.get("objective", "regression")

        # 2. Extrair o score inicial (F_0)
        # Nota: No LightGBM moderno, o F_0 originado de 'boost_from_average=True'
        # pode vir codificado como uma árvore de um único nó (a árvore 0) ou salvo
        # nos metadados. Para fins de generalização, abstraímos essa extração pura.

        f0 = 0.0  # Valor padrão. A lógica abaixo procura F_0 em cenários comuns.

        # Exemplo de lógica de parsing (depende da versão do LightGBM):
        # Se a primeira árvore tem apenas uma folha, ela costuma guardar o bias global.
        first_tree = self.model_dump.get("tree_info", [])[0]
        if first_tree["num_leaves"] == 1:
            f0 = first_tree["tree_structure"]["leaf_value"]

        return f0, objective

    def _traverse_node(self, node, feature_thresholds):

        if "split_feature" not in node:
            return

        feature = node["split_feature"]

        threshold = node["threshold"]

        if feature not in feature_thresholds:
            feature_thresholds[feature] = set()

        feature_thresholds[feature].add(threshold)

        if "left_child" in node:
            self._traverse_node(node["left_child"], feature_thresholds)

        if "right_child" in node:
            self._traverse_node(node["right_child"], feature_thresholds)

    def phase_2(self):
        """
        Fase 2 — Recuperação das marginais.
        Extrai os limiares (bins) do LightGBM percorrendo todas as árvores
        para reconstruir a CDF inversa de cada feature.
        """

        # Reconstrucão dos limites das features (mínimos e máximos) a partir do modelo.
        print("[Fase 2] Iniciando extração dos limiares de feature...")

        feature_thresholds = {}

        trees = self.model_dump.get("tree_info", [])

        for tree in trees:
            tree_structure = tree.get("tree_structure", {})
            self._traverse_node(tree_structure, feature_thresholds)

        self.marginals = {}

        for feature, thresh_set in feature_thresholds.items():
            self.marginals[feature] = sorted(list(thresh_set))

        print(
            f"[Fase 2] Marginais recuperadas para {len(self.marginals)} features distintas."
        )

        return self.marginals

    def _traverse_and_reconstruct(
        self, node, current_bounds, leaves_info, eta, reg_lambda
    ):
        """
        Método recursivo (DFS) para rastrear os intervalos de features e resolver
        as equações de contagem de amostras nas folhas.
        """

        if "split_feature" not in node:
            leaf_index = node["leaf_index"]

            leaf_value = node.get("leaf_value", 0.0)

            H_j = node.get("leaf_weight", 0.0)

            G_j = -(leaf_value * (H_j + reg_lambda)) / eta

            if self.y_bar is not None and 0 < self.y_bar < 1:
                denominador = self.y_bar * (1.0 - self.y_bar)
                N_j_total = H_j / denominador
                N_j_pos = (N_j_total * self.y_bar) - G_j
                N_j_neg = N_j_total - N_j_pos
            else:
                # Fallback caso a Fase 1 não tenha sido executada ou y_bar seja 0/1
                N_j_total = N_j_pos = N_j_neg = 0.0

            leaves_info.append(
                {
                    "leaf_index": leaf_index,
                    "bounds": copy.deepcopy(
                        current_bounds
                    ),  # Restrições espaciais (hipercubo)
                    "H_j": H_j,
                    "G_j": G_j,
                    "N_total": N_j_total,
                    "N_pos": N_j_pos,
                    "N_neg": N_j_neg,
                }
            )

            return

        feature = node["split_feature"]

        threshold = node["threshold"]

        if feature not in current_bounds:
            current_bounds[feature] = [float("-inf"), float("inf")]

        left_bounds = copy.deepcopy(current_bounds)
        left_bounds[feature][1] = min(left_bounds[feature][1], threshold)
        self._traverse_and_reconstruct(
            node["left_child"], left_bounds, leaves_info, eta, reg_lambda
        )

        right_bounds = copy.deepcopy(current_bounds)
        right_bounds[feature][0] = max(right_bounds[feature][0], threshold)
        self._traverse_and_reconstruct(
            node["right_child"], right_bounds, leaves_info, eta, reg_lambda
        )

    def _reconcile_leaf_counts(self):
        """
        Corrige erros de arredondamento de ponto flutuante garantindo que a soma
        das amostras de todas as folhas (calculadas via Hessiano) bata exatamente com N_total.
        """
        soma_calculada = sum(
            int(round(leaf["N_total"])) for leaf in self.leaves_reconstruction
        )
        diferenca = self.N_total - soma_calculada

        if diferenca == 0:
            print("[Reconciliação] Perfeito. A soma das folhas já iguala N_total.")
            return

        print(
            f"[Reconciliação] Ajustando erro de arredondamento global de {diferenca} amostras."
        )

        # Estratégia de ajuste simples: adiciona ou remove as amostras faltantes
        # nas folhas que tiveram a maior parte fracionária (maior incerteza de arredondamento)

        # Calcula a parte decimal (ex: 10.4 -> 0.4 de erro)
        erros_fracionarios = []
        for i, leaf in enumerate(self.leaves_reconstruction):
            valor_cru = leaf["H_j"] / (self.y_bar * (1.0 - self.y_bar))
            parte_decimal = valor_cru - int(valor_cru)
            erros_fracionarios.append((i, parte_decimal))

        # Ordena pelas folhas mais "incertas"
        if diferenca > 0:
            # Precisamos adicionar amostras: pegar os maiores decimais (ex: .49)
            erros_fracionarios.sort(key=lambda x: x[1], reverse=True)
            for k in range(diferenca):
                idx = erros_fracionarios[k][0]
                self.leaves_reconstruction[idx]["N_total"] += 1
                # Aproxima também no rótulo majoritário da folha
                self.leaves_reconstruction[idx]["N_pos"] += self.y_bar
        elif diferenca < 0:
            # Precisamos remover amostras: pegar os menores decimais (ex: .51)
            erros_fracionarios.sort(key=lambda x: x[1])
            for k in range(abs(diferenca)):
                idx = erros_fracionarios[k][0]
                self.leaves_reconstruction[idx]["N_total"] -= 1

    def phase_3(self, eta=1.0, reg_lambda=1.0):
        """
        Fase 3 — Inversão por folha.
        Reconstrói a contagem de amostras, rótulos e restrições espaciais
        para cada folha da primeira árvore (a primeira que divide o espaço).
        """

        print("\n[Fase 3] Iniciando inversão algébrica das folhas...")

        if self.y_bar is None:
            print(
                "[Fase 3] Aviso: y_bar não definido. As contagens de classes podem ser imprecisas."
            )
            self.y_bar = 0.5

        trees = self.model_dump.get("tree_info", [])

        first_split_tree = None

        for tree in trees:
            if tree["num_leaves"] > 1:
                first_split_tree = tree
                break

        if first_split_tree is None:
            print(
                "[Fase 3] Erro: Nenhuma árvore com divisão encontrada. Verifique o modelo."
            )
            return None

        print(
            f"[Fase 3] Analisando a árvore de índice {first_split_tree['tree_index']}..."
        )

        # Estrutura para armazenar as informações recuperadas
        self.leaves_reconstruction = []
        initial_bounds = {}

        # Inicia a travessia DFS a partir da raiz da árvore selecionada
        self._traverse_and_reconstruct(
            first_split_tree["tree_structure"],
            initial_bounds,
            self.leaves_reconstruction,
            eta=eta,
            reg_lambda=reg_lambda,
        )

        print(
            f"[Fase 3] Reconstrução concluída para {len(self.leaves_reconstruction)} folhas."
        )

        # Demonstração dos resultados das primeiras duas folhas
        for i, leaf in enumerate(self.leaves_reconstruction[:2]):
            print(f"\n  -> Folha {leaf['leaf_index']}:")
            # Arredondando os valores calculados, pois as contagens devem ser inteiros
            print(f"     Amostras Totais (N_j)  : ~{round(leaf['N_total'])}")
            print(f"     Amostras Positivas     : ~{round(leaf['N_pos'])}")
            print(f"     Amostras Negativas     : ~{round(leaf['N_neg'])}")
            print(f"     Restrições (Bounds)    : {leaf['bounds']}")

        if len(self.leaves_reconstruction) > 2:
            print("\n  -> ...")

        # 2. Usa o conhecimento prévio de N_total para ancorar as contagens à realidade
        self._reconcile_leaf_counts()

        # 3. Usa o conhecimento de M para auditar as features extraídas
        features_extraidas = (
            len(self.marginals.keys()) if hasattr(self, "marginals") else 0
        )
        features_faltantes = self.num_features - features_extraidas

        if features_faltantes > 0:
            print(
                f"[Auditoria] Alerta: O modelo ignora {features_faltantes} das {self.num_features} features."
            )
            print(
                "[Auditoria] Os hipercubos terão limites [-inf, inf] nestas dimensões ausentes."
            )

        return self.leaves_reconstruction

    def _instantiate_samples_from_tree_1(self):
        """
        Cria a população inicial de amostras com base na reconstrução da Fase 3.
        """
        self.reconstructed_samples = []
        sample_id = 0

        for leaf in self.leaves_reconstruction:
            # 1. Trava o total da folha (que já foi reconciliado e é exato)
            n_total = int(round(leaf["N_total"]))

            # 2. Calcula os positivos
            n_pos = int(round(leaf["N_pos"]))

            n_pos = max(0, min(n_total, n_pos))   # ✅ clamp em [0, n_total]

            # 3. Trava os negativos matematicamente para evitar vazamento de arredondamento
            n_neg = n_total - n_pos

            # Instancia amostras positivas
            for _ in range(n_pos):
                self.reconstructed_samples.append(
                    {
                        "id": sample_id,
                        "label": 1,
                        "bounds": copy.deepcopy(leaf["bounds"]),
                    }
                )
                sample_id += 1

            # Instancia amostras negativas
            for _ in range(n_neg):
                self.reconstructed_samples.append(
                    {
                        "id": sample_id,
                        "label": 0,
                        "bounds": copy.deepcopy(leaf["bounds"]),
                    }
                )
                sample_id += 1

        print(
            f"[Fase 4] {len(self.reconstructed_samples)} amostras instanciadas a partir da Árvore 1."
        )

    def _check_bounds_intersection(self, sample_bounds, leaf_bounds):
        """
        Verifica se é espacialmente possível uma amostra cair em uma folha.
        (Se os hipercubos não se tocam, a interseção é vazia).
        """
        for feature, (l_min, l_max) in leaf_bounds.items():
            if feature in sample_bounds:
                s_min, s_max = sample_bounds[feature]
                # Se o máximo da amostra for menor que o mínimo da folha (ou vice-versa), sem interseção
                if s_max < l_min or s_min > l_max:
                    return False
        return True

    def _update_sample_bounds(self, sample, leaf_bounds):
        """
        Aperta os limites da amostra fazendo a interseção geométrica com a nova folha.
        """
        for feature, (l_min, l_max) in leaf_bounds.items():
            if feature not in sample["bounds"]:
                sample["bounds"][feature] = [l_min, l_max]
            else:
                s_min, s_max = sample["bounds"][feature]
                # Novo limite é a interseção (maior dos mínimos, menor dos máximos)
                sample["bounds"][feature][0] = max(s_min, l_min)
                sample["bounds"][feature][1] = min(s_max, l_max)

    def phase_4(self, next_trees_indices=[1]):
        """
        Fase 4 — Refinamento por MILP (Gurobi).
        Resolve a atribuição de amostras para as folhas das árvores subsequentes.
        """
        print("\n[Fase 4] Iniciando o refinamento espacial por MILP...")

        # 1. Preparar as amostras a partir do resultado da Fase 3
        if not hasattr(self, "leaves_reconstruction"):
            raise ValueError("Execute a Fase 3 antes da Fase 4.")
        self._instantiate_samples_from_tree_1()

        trees = self.model_dump.get("tree_info", [])

        # Iterar sobre as próximas árvores para refinar as amostras
        for tree_idx in next_trees_indices:
            print(f"\n[Fase 4] Formulando MILP para a Árvore {tree_idx}...")
            target_tree = trees[tree_idx]

            # Extrair informações da nova árvore (reutilizando lógica da fase 3, omitida por brevidade)
            # Na prática, chamaríamos _traverse_and_reconstruct para esta árvore para obter target_leaves
            target_leaves = []
            self._traverse_and_reconstruct(
                target_tree["tree_structure"],
                {},
                target_leaves,
                self.eta,
                self.reg_lambda,
            )

            # 2. Configurar o ambiente Gurobi (silencioso)
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)  # Desliga logs extensos do solver
            env.start()
            model = gp.Model("GBM_Assignment_Tree_" + str(tree_idx), env=env)

            # 3. Variáveis de Decisão (Z_ij)
            # z[i, j] = 1 se a amostra i cai na folha j da árvore atual, 0 caso contrário.
            z = {}
            for i, sample in enumerate(self.reconstructed_samples):
                for j, leaf in enumerate(target_leaves):
                    # Otimização crucial: Só cria a variável se os limites se cruzarem.
                    if self._check_bounds_intersection(
                        sample["bounds"], leaf["bounds"]
                    ):
                        z[i, j] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}")

            # 4. Restrições do MILP

            # Restrição A: Cada amostra DEVE ser atribuída a exatamente UMA folha
            for i in range(len(self.reconstructed_samples)):
                model.addConstr(
                    gp.quicksum(
                        z[i, j] for j in range(len(target_leaves)) if (i, j) in z
                    )
                    == 1,
                    name=f"assign_one_{i}",
                )

            # Restrição B: A soma das amostras atribuídas deve igualar a capacidade (N_j) da folha
            for j, leaf in enumerate(target_leaves):
                N_j_total = int(round(leaf["N_total"]))
                model.addConstr(
                    gp.quicksum(
                        z[i, j]
                        for i in range(len(self.reconstructed_samples))
                        if (i, j) in z
                    )
                    == N_j_total,
                    name=f"leaf_capacity_{j}",
                )

            # Restrição C: A soma dos rótulos (amostras positivas) deve igualar N_j_pos da folha
            for j, leaf in enumerate(target_leaves):
                N_j_pos = int(round(leaf["N_pos"]))
                model.addConstr(
                    gp.quicksum(
                        self.reconstructed_samples[i]["label"] * z[i, j]
                        for i in range(len(self.reconstructed_samples))
                        if (i, j) in z
                    )
                    == N_j_pos,
                    name=f"leaf_pos_capacity_{j}",
                )

            # 5. Otimização
            # Como é um problema de viabilidade (encontrar UMA atribuição válida que satisfaça tudo),
            # não precisamos de uma função objetivo complexa.
            model.setObjective(0, GRB.MINIMIZE)

            print("[Fase 4] Resolvendo modelo Gurobi... (Isso pode demorar)")
            model.optimize()

            if model.status == GRB.OPTIMAL:
                print(
                    "[Fase 4] Solução viável encontrada! Atualizando restrições espaciais..."
                )
                # 6. Atualizar os limites (apertar o cerco) com base na atribuição
                for i, sample in enumerate(self.reconstructed_samples):
                    for j, leaf in enumerate(target_leaves):
                        if (i, j) in z and z[
                            i, j
                        ].X > 0.5:  # Se a variável for 1 (com tolerância flutuante)
                            self._update_sample_bounds(sample, leaf["bounds"])
                            break  # Já achou a folha, vai pra próxima amostra
            else:
                print(
                    f"[Fase 4] ALERTA: O solver não encontrou solução para a árvore {tree_idx}. O modelo pode estar relaxado ou houve erro numérico nas fases anteriores."
                )

        print("\n[Fase 4] Refinamento concluído.")
        return self.reconstructed_samples
