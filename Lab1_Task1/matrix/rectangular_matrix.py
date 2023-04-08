import numpy as np

class RandomMatrix:

    def __init__(self, numero_filas:int, numero_columnas:int):
        self.numero_filas = numero_filas
        self.numero_columnas = numero_columnas
        self.matrix = np.random.randint(100,size =(self.numero_filas, self.numero_columnas))

    def __str__(self) -> str:
        return f'A random matrix  of size  {self.numero_filas}x{self.numero_columnas} has been generated'
    
    def set_matrix(self):
        """" Create a random maxtrix """
        self.matrix = np.random.randint(100,size =(self.numero_filas, self.numero_columnas))
    
    def transpose(self):
        """" This method calculate yhe transpose of matrix"""
        return np.transpose(self.matrix)
    
    def rank(self):
        """" Return rank of matrix"""
        return np.linalg.matrix_rank(self.matrix)
    
    def trace(self):
        """" Return trace of matrix"""
        return np.matrix.trace(self.matrix)
    
    def determinant(self):
        """" Calculate matrix's determinant"""
        try: 
            if self.matrix.shape[0] == self.matrix.shape[1]:
                return np.linalg.det(self.matrix)
            else :
                raise Exception('Cannot calculate the determinan of the matrix')
        except Exception as e :
            print(f'An exception occurred: {e}')

    def inverse(self):
        """"if is a square matrix compute the (multiplicative) inverse of a matrix. in other cases Calculate the generalized inverse of a matrix using its singular-value decomposition
            (SVD) and including all large singular values."""
        try:
            return np.linalg.inv(self.matrix)
        except:
            return np.linalg.pinv(self.matrix)

