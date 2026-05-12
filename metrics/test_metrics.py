import unittest
import numpy as np

# Supondo que o seu arquivo com as funções se chame 'metrics.py'
# Caso tenha outro nome, altere o import abaixo:
from metrics import calcular_ra_timberstrike

class TestTimberStrikeRA(unittest.TestCase):

    def setUp(self):
        """
        Configuração inicial executada antes de cada teste.
        Criamos um dataset "real" simples (ground truth).
        Colunas: [Idade (contínua), Renda (contínua), Gênero (categórica)]
        """
        self.indices_categoricos = [2]
        self.X_real = np.array([
            [25.0, 3000.0, 1], # Linha 0
            [40.0, 5000.0, 0], # Linha 1
            [60.0, 8000.0, 1]  # Linha 2
        ])
        
        # O multiplicador padrão do artigo é 0.319
        self.tol = 0.319

    def test_reconstrucao_perfeita(self):
        """
        Cenário 1: O atacante reconstrói exatamente os mesmos dados.
        O RA deve ser 100%.
        """
        X_rec = self.X_real.copy()
        ra = calcular_ra_timberstrike(self.X_real, X_rec, self.indices_categoricos, self.tol)
        
        self.assertAlmostEqual(ra, 100.0, places=2, msg="RA deveria ser 100% para dados idênticos.")

    def test_reconstrucao_embaralhada(self):
        """
        Cenário 2: O atacante reconstrói dados perfeitos, mas fora de ordem.
        O Algoritmo Húngaro deve pareá-los perfeitamente, resultando em 100%.
        """
        X_rec = np.array([
            [60.0, 8000.0, 1], # Era a Linha 2
            [25.0, 3000.0, 1], # Era a Linha 0
            [40.0, 5000.0, 0]  # Era a Linha 1
        ])
        ra = calcular_ra_timberstrike(self.X_real, X_rec, self.indices_categoricos, self.tol)
        
        self.assertAlmostEqual(ra, 100.0, places=2, msg="RA deveria ser 100% mesmo com os dados embaralhados.")

    def test_reconstrucao_com_ruido(self):
        """
        Cenário 3: O atacante introduz erros na reconstrução.
        Vamos calcular o erro esperado manualmente.
        
        Desvios padrão (aproximados): Idade: ~14.33 (Tol: ~4.57), Renda: ~2054 (Tol: ~655)
        """
        X_rec = np.array([
            # Linha 0: Todos certos. Erro = 0. Acurácia = 1.0 (3/3)
            [25.0, 3000.0, 1], 
            
            # Linha 1: Idade OK, Renda OK (+100 está dentro da tol de ~655), Gênero ERRADO. 
            # Erro = 1 categórico. Acurácia = 0.666 (2/3)
            [40.0, 5100.0, 1], 
            
            # Linha 2: Idade ERRADA (-10 é > tol de 4.57), Renda OK, Gênero OK.
            # Erro = 1 contínuo. Acurácia = 0.666 (2/3)
            [50.0, 8000.0, 1]  
        ])
        
        ra = calcular_ra_timberstrike(self.X_real, X_rec, self.indices_categoricos, self.tol)
        
        # Acurácia média esperada: (1.0 + 0.666... + 0.666...) / 3 = 0.7777... => 77.77%
        self.assertAlmostEqual(ra, 77.78, places=1, msg="RA calculou penalidades incorretas para features com erro.")

    def test_tamanhos_diferentes(self):
        """
        Cenário 4: O atacante reconstruiu um número diferente de linhas.
        A função deve selecionar um subset aleatório (tamanho mínimo) e calcular sem quebrar.
        """
        X_rec_maior = np.array([
            [25.0, 3000.0, 1],
            [40.0, 5000.0, 0],
            [60.0, 8000.0, 1],
            [99.0, 9999.0, 0] # Linha extra/inventada
        ])
        
        try:
            ra = calcular_ra_timberstrike(self.X_real, X_rec_maior, self.indices_categoricos, self.tol)
            rodou_sem_erro = True
        except Exception as e:
            rodou_sem_erro = False
            
        self.assertTrue(rodou_sem_erro, "A função deveria rodar sem erros mesmo com matrizes de tamanhos diferentes.")
        # Como as matrizes têm tamanhos diferentes, o algoritmo vai parear as 3 linhas reais com 3 das 4 reconstruídas
        # O RA não será 100%, mas deve retornar um número float válido.
        self.assertIsInstance(ra, float)

if __name__ == '__main__':
    # Roda os testes no terminal
    unittest.main(verbosity=2)