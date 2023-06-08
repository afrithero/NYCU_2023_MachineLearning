import copy
import sys
import matplotlib.pyplot as plt

'''
@Description: 定義 Matrix 類別已實作矩陣運算
@Field: m 表示 row num, n 表示 col num, entities 包含二維陣列所有元素
@Method: 
    static method:
        -   create_identity_matrix: 建立單位矩陣
        -   add_matrix: 矩陣相加
        -   multiply_matrix: 矩陣相乘
        -   multiply_scalar: 矩陣純量相乘
    private method:
        -   LU_decomposition: 求反矩陣時需利用 LU 分解 LUA^-1 = I
        -   forward_substitution: Ly = I 求 y, y = UA^-1
        -   back_substitution: UA^-1 = y 求 A^-1
    publice method:
        -   inverse: 求反矩陣
        -   transpose: 求轉置矩陣
'''
class Matrix:
    def __init__(self, matrix):
        self.m = len(matrix)
        self.n = len(matrix[0])
        self.entities = matrix

    @staticmethod
    def create_identity_matrix(dim):
        identity_matrix = [[0]*i+[1]+[0]*(dim-i-1) for i in range(dim)]
        return Matrix(identity_matrix)
    
    @staticmethod
    def add_matrix(matrix_1, matrix_2):
        result = [[0 for col in range(matrix_1.n)] for row in range(matrix_1.m)]
        for i in range(matrix_1.m):
            for j in range(matrix_2.n):
                result[i][j] = matrix_1.entities[i][j] + matrix_2.entities[i][j]
        return Matrix(result)

    @staticmethod
    def multiply_matrix(matrix_1, matrix_2):
        result = [[0 for col in range(matrix_2.n)] for row in range(matrix_1.m)]
        for i in range(matrix_1.m):
            for j in range(matrix_2.n):
                for k in range(matrix_2.m):
                    result[i][j] += matrix_1.entities[i][k] * matrix_2.entities[k][j]
        return Matrix(result)
    
    @staticmethod
    def multiply_scalar(matrix, scalar):
        for i in range(matrix.m):
            for j in range(matrix.n):
                matrix.entities[i][j] *= scalar
        return matrix

    def __LU_decomposition(self):
        upper_matrix = copy.deepcopy(self)
        lower_matrix = [[0]*i+[1]+[0]*(self.m-i-1) for i in range(self.m)]
        for i in range(self.m):
            for j in range(i+1,self.m):
                ratio = upper_matrix.entities[j][i] / upper_matrix.entities[i][i]
                lower_matrix[j][i] = ratio
                for k in range(self.m):
                    upper_matrix.entities[j][k] -= ratio * upper_matrix.entities[i][k]
        return Matrix(lower_matrix), upper_matrix
    
    def __forward_substitution(self):
        y = Matrix([[0 for col in range(self.m)] for row in range(self.m)])
        for i in range(y.n): # 第一層是 y matrix 的 col
            for j in range(y.m): # 第二層是 y matrix 的 row
                # temp 是 Ly = I 的 I
                if i == j:
                    temp = 1
                else:
                    temp = 0
                for k in range(0,j): #第三層要回去看 L 的其他變數
                    temp -= self.entities[j][k] * y.entities[k][i] # I 減去已經求好的解 -> ex. y 第一個位置乘上 Lowermatrix 第二列的第一個係數， I 再減掉這個解即得 y 第二個位置的值
                y.entities[j][i] = temp
        return y
    
    def __back_substitution(self, y_matrix): # y 在等號右邊
        x = Matrix([[0 for col in range(self.m)] for row in range(self.m)])
        for i in range(self.n-1, -1, -1): # 第一層是 Upper matrix 的 col
            for j in range(self.m-1, -1, -1): # 第二層是 Upper matrix 的 row
                temp = y_matrix.entities[j][i]
                for k in range(j+1, self.m): # 要加 1，不然 j = 2 的時候會執行一次
                    temp -= self.entities[j][k] * x.entities[k][i] # y_matrix 的值減去已經求好的解
                x.entities[j][i] = temp / self.entities[j][j] # 答案除以係數（upper matrix）後即得解
        return x

    def inverse(self):
        lower_matrix, upper_matrix = self.__LU_decomposition()
        y_matrix = lower_matrix.__forward_substitution()
        x_matrix = upper_matrix.__back_substitution(y_matrix)
        return x_matrix

    def transpose(self):
        matrix_transposed = [[0 for col in range(self.m)] for row in range(self.n)]
        for i in range(self.m):
            for j in range(self.n):
                matrix_transposed[j][i] = self.entities[i][j]
        return Matrix(matrix_transposed)


def least_square_solution(design_matrix, ground_truth, regularization):
    design_matrix_transposed = design_matrix.transpose()
    result_matrix = Matrix.multiply_matrix(design_matrix_transposed,design_matrix)
    if regularization != 0:
        identity_matrix = Matrix.create_identity_matrix(result_matrix.m)
        regularization_matrix = Matrix.multiply_scalar(identity_matrix, regularization)
        result_matrix = Matrix.add_matrix(result_matrix, regularization_matrix)
    result_matrix_inverse = result_matrix.inverse()
    temp_matrix = Matrix.multiply_matrix(result_matrix_inverse, design_matrix_transposed)
    solution = []
    for i in range(temp_matrix.m):
        temp = 0
        for j in range(temp_matrix.n):
            temp += temp_matrix.entities[i][j] * ground_truth[j]
        solution.append(temp)
    return solution

def newtons_method_solution(desgin_matrix, ground_truth):
    design_matrix_transposed = design_matrix.transpose()
    result_matrix = Matrix.multiply_matrix(design_matrix_transposed,design_matrix)
    result_matrix_inverse = result_matrix.inverse()
    temp_matrix = Matrix.multiply_matrix(result_matrix_inverse, design_matrix_transposed)
    solution = []
    for i in range(temp_matrix.m):
        temp = 0
        for j in range(temp_matrix.n):
            temp += temp_matrix.entities[i][j] * ground_truth[j]
        solution.append(temp)
    return solution

def calculate_error(design_matrix, solution, ground_truth):
    prediction = []
    for i in range(design_matrix.m):
        temp = 0
        for j in range(design_matrix.n):
            temp += design_matrix.entities[i][j] * solution[j]
        prediction.append(temp)
    error_list = [x - y for x, y in zip(prediction, ground_truth)]
    error = 0
    for k in range(len(error_list)):
        error += error_list[k] ** 2
    return prediction, error

def generate_equation(order, solution):
    x_list = []
    for i in range(order):
        x_list.append("X^{}".format(i))

    x_list.reverse()

    for i in range(len(x_list)):
        if i == len(x_list) - 1:
            x_list[i] = str(solution[i])
        else:
            x_list[i] = str(solution[i]) +x_list[i]
    
    equation = ""
    
    for j in range(len(x_list)):
        if j == 0:
            equation += x_list[j]
        else:
            if x_list[j][0] == "-":
                equation += x_list[j]
            else:
                equation += "+" + x_list[j]
    return equation


order = int(sys.argv[1])
regularization = int(sys.argv[2])
file_path = str(sys.argv[3])

with open(file_path, 'r') as file:
    data_points = []
    for line in file:
        x, y = map(float, line.rstrip('\n').split(","))
        data_points.append((x,y))

input_value = []
for i in range(len(data_points)):
    input_value.append(data_points[i][0])

ground_truth = []
for i in range(len(data_points)):
    ground_truth.append(data_points[i][1])

design_matrix = []
for i in range(len(data_points)):
    design_matrix_row = []
    for j in range(order-1,-1,-1):
        design_matrix_row.append(data_points[i][0]**(j))
    design_matrix.append(design_matrix_row)

design_matrix = Matrix(design_matrix)

least_square_solution = least_square_solution(design_matrix, ground_truth, regularization)
least_square_prediction, least_square_error = calculate_error(design_matrix,least_square_solution,ground_truth)

newtons_method_solution = newtons_method_solution(design_matrix, ground_truth)
newtons_method_prediction, newtons_method_error = calculate_error(design_matrix,newtons_method_solution,ground_truth)

print("LSE:")
print("Fitting line: " + generate_equation(order, least_square_solution))
print("Total error: {}".format(least_square_error))

print("Newton's Method:")
print("Fitting line: " + generate_equation(order, newtons_method_solution))
print("Total error: {}".format(newtons_method_error))

figure = plt.figure()
figure.suptitle('Machine Learning HW1', fontsize=16)
figure.add_subplot(211)
plt.scatter(input_value,ground_truth,linewidths=0.1)
plt.plot(input_value,least_square_prediction,c="red")
plt.title("LSE")
figure.add_subplot(212)
plt.scatter(input_value,ground_truth,linewidths=0.1)
plt.plot(input_value,newtons_method_prediction,c="red")
plt.title("Newton's Method")
plt.tight_layout()
plt.show()


