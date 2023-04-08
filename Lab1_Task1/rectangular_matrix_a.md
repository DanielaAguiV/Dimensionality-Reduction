# Task 1

In this notebook you can see the solution for task 1. The class Matrix generate a rectangular matrix A with the shape of your preference. 
You can see the class in the carpet matrix and an example in *rectangular_matrix_a.ipynb*

**Note:** The determinant is only definite to squared matrix

**Note:** When not is posible calculate the inverse we calculate pseudoinverse

Let $A$ be an $n \times m$ matrix, then the reletationship between $AA^T$ and $A^TA$, by theorem it's non-zero eingvalues are the same furthermore if $v$ is an eigenvector of $AA^T$ corresponding to the eigenvalue $\lambda$ the $Av$ is an eigenvector of $A^TA$ correspondinng to the same $\lambda$, The eigenvectors of $AA^T$ and $A^TA$ form an orthonormal basis for $R^n$ and $R^m$ respectively: The eigenvectors of $AA^T$ form an orthonormal basis for the column space of A, while the eigenvectors of $A^TA$ form an orthonormal basis for the row space of $A$