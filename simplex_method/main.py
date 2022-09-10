from scipy import linalg
import numpy as np

"""
standard form:
    max 3x+5y
    s.t x+ 0+ s1       =4
        0+2y+    s2    =12
        3x+2y+       s3=18
        x,y       >=0
abstract form:
    max c^t x
    s.t. ax=b
         x>=0        

steps for tableu simplex method
1.basic variables = {non-slacks}, build tableu
2.make non-slack=0, solve bf solution 1st
3.optimal test based on the indicator>=0
while not all indicator >=0:
    4.find the minimum indicator,entering variables , pivot column 
      minimum ratio test, pivot_row, leaving variables
      update the basic variables list.
    5.gaussian termination using the pivot row on other rows including target, 
      find bf solution,
"""


class Solver:
    def __init__(self, target, consts):
        self.target = target
        self.consts = consts

    def show_solution(self, x, basic_v):
        solution = [0 for i in range(self.consts.shape[1] - 1)]
        for i, v in enumerate(basic_v):
            solution[v] = x[i]
        return solution

    def solve(self):
        consts = self.consts
        target = self.target
        import copy
        target_fin = copy.deepcopy(target)
        slack_n = consts.shape[0]
        none_basic_n = target.shape[1] - slack_n - 1
        basic_vs = [i for i in range(none_basic_n, none_basic_n + slack_n)]

        # we now have ax=b where a,b all exist in const matrix
        # get bf solution
        x = linalg.solve(consts[:, basic_vs], consts[:, -1])
        solu = self.show_solution(x, basic_vs)
        print("ax=b : \n", consts, "\ntarget :\n", target_fin, "\nsolution is : \n", solu)
        print("\n-----------------------------\n")
        # optimal test

        def is_optimal(tar, basic_v, x):
            solution = self.show_solution(x, basic_v)
            for each in list(tar[0, :-1]):
                if each < 0:
                    print("solution not optimal", solution)
                    return 0
            print("solution  optimal", solution)
            # generating solution
            final_z = np.dot(np.array(solution[:none_basic_n]), np.transpose(-target_fin[0, :none_basic_n]))
            print("z : ", final_z, solution[:none_basic_n], -target_fin[0, :none_basic_n])
            return 1


        while not is_optimal(target,basic_vs,x):
            print("\n-----------------------------\n")
            # find pivot column
            li = list(target[0, :-1])
            pivot_column_n = li.index(min(li))
            print("indicators", li, "pivot column index", pivot_column_n)
            # find pivot row via minimum ratio test.
            # (MRT finds the most restrictive constrains on our way,
            # the way chosen by min indicator aka max increase variable of z)
            li = [consts[i, -1]/consts[i, pivot_column_n] if consts[i, pivot_column_n] != 0 else float("inf") for i in range(consts.shape[0])]
            for i in range(consts.shape[0]):
                print("pivot_num", consts[i, pivot_column_n], "\  =", consts[i, -1])
            pivot_row_n = li.index(min(li))
            print("ratios ,", li, "pivot row index", pivot_row_n)
            # update basic_variables, gaussian termination
            basic_vs[pivot_row_n] = pivot_column_n  # index of leaving variables = pivot_row_n; entering_v = pivot_column_n
            ## gaussian termination for ax=b and target
            for i in range(slack_n):
                if i == pivot_row_n:
                    continue
                consts[i, :] = consts[i, :] - (consts[i, pivot_column_n] / consts[pivot_row_n, pivot_column_n]) * consts[pivot_row_n, :]
            target = target - (target[0, pivot_column_n]/consts[pivot_row_n, pivot_column_n])*consts[pivot_row_n, :]
            print("target", target)
            print("basic_vs", basic_vs)
            # bf solution
            x = linalg.solve(consts[:, basic_vs], consts[:, -1])


if __name__=="__main__":
    target = np.array(
        [[-3, -5, 0, 0, 0, 0]]
    )
    consts = np.array(
        [[1, 0, 1, 0, 0, 4],
         [0, 2, 0, 1, 0, 12],
         [3, 2, 0, 0, 1, 18]]
    )  # the last num is '='
    pro = Solver(target, consts)
    pro.solve()

