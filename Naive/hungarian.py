import numpy as np


def min_zero_row(zero_mat, mark_zero):
    sum_rows = np.sum(zero_mat, axis=0)
    sum_rows[np.where(sum_rows == 0)] = zero_mat.shape[0]
    min_row = np.argmin(sum_rows)
    # quit()
    # # quit()
    zero_index = np.where(zero_mat[:,min_row] == True)[0][0]
    zero_mat[:,min_row] = False
    zero_mat[zero_index,:] = False
    mark_zero.append((min_row, zero_index))

def mark_matrix(mat):
    zeros = mat == 0
    # # quit()
    zero_copy = zeros.copy()

    marked_zero = []
    while True in zero_copy:
        min_zero_row(zero_copy, marked_zero)


    marked_zero_row = []
    marked_zero_col = []

    for i in range(len(marked_zero)):
        marked_zero_row.append(marked_zero[i][0])
        marked_zero_col.append(marked_zero[i][1])

    # Find any non-marked rows
    non_marked_rows = list(set(range(mat.shape[1])) - set(marked_zero_row))

    marked_cols = []
    check_switch = True
    while check_switch:
        check_switch = False
        for i in range(0, len(non_marked_rows)):
            row_array = zeros[:,non_marked_rows[i]]
            for j in range(row_array.shape[0]):
                if row_array[j] == True and j not in marked_cols:
                    marked_cols.append(j)
                    check_switch = True
        for i in range(len(marked_zero)):
            row_num, col_num = marked_zero[i][0], marked_zero[i][1]
            if row_num not in non_marked_rows and col_num in marked_cols:
                non_marked_rows.append(row_num)
                check_switch = True
    marked_rows = list(set(range(mat.shape[1])) - set(non_marked_rows))
    return (marked_zero, marked_rows, marked_cols)

def adjust_matrix(mat, cover_rows, cover_cols):
    mat = mat.copy()
    non_zero_element = []
    for row in range(len(mat)):
        if row not in cover_rows:
            for i in range(len(mat[row])):
                if i not in cover_cols:
                    non_zero_element.append(mat[row, i])
    min_num = min(non_zero_element)

    for row in range(len(mat)):
        if row not in cover_rows:
            for i in range(len(mat[row])):
                if i not in cover_cols:
                    mat[row, i] = mat[row, i] - min_num

    for row in cover_rows:
        for col in cover_cols:
            mat[row, col] = mat[row, col] + min_num

    return mat


def hungarian_algorithm_method(X):
    X = X.copy()

    # Step 1: subtract every row from its minimum
    print(X.shape)
    X = (X.T - np.min(X, axis=1).T).T
    # for row in range(X.shape[0]):
        # print(np.where(test[row,:] == True))

    # quit()
    
    # Step 2: subtract every column from its minimum
    X = X - np.min(X, axis=0)
    zero_count = 0
    while zero_count < X.shape[1]:
        ans_pos, marked_cols, marked_rows = mark_matrix(X)
        zero_count = len(marked_rows) + len(marked_cols)

        if zero_count < X.shape[1]:
            X = adjust_matrix(X, marked_rows, marked_cols)

    a = [None, None, None]
    for i, c in ans_pos:
        a[i] = c
    return a


def brute_algorithm(X):
    x_arr = X[:,0]
    y_arr = X[:,1]
    z_arr = X[:,2]

    cost_grid = np.meshgrid(x_arr, y_arr, z_arr, copy=True)
    for g in range(len(cost_grid)):
        for i in range(X.shape[1]):
            cost_grid[g][i,i,i] = np.inf
            cost_grid[g][i,i,:] = np.inf
            cost_grid[g][:, i, i] = np.inf
            cost_grid[g][i, :, i] = np.inf
        cost_grid[g] = cost_grid[g]
    sum_grids = None
    for g in cost_grid:
        if sum_grids is None:
            sum_grids = g
        else:
            sum_grids = sum_grids + g
    x_choices, y_choices, z_choices = np.where(sum_grids == np.min(sum_grids))
    # print(np.where(sum_grids == np.min(sum_grids)))
    # print(sum_grids[x_choices[0], y_choices[0], z_choices[0]])

    # quit()
    x_choice, y_choice, z_choice = x_choices[0], y_choices[0], z_choices[0]
    
    return [x_choice, y_choice, z_choice]


if __name__ == '__main__':
    test_array = np.array([[ 2.5 ,  3.  ,  3.25,  6.  ,  8.  ,  9.4, 1.   ], [ 3.  ,  4.  ,  2.  , 10.  , 19.  , 53.  , 29.  ], [ 8.  ,  4.  ,  32.  , 29.  , 9.  , 5.  , 83.  ]]).T
    sol = hungarian_algorithm_method(test_array)
    b = brute_algorithm(test_array)
    print(sol, sum([test_array[x, i] for i, x in enumerate(sol)]))
    print(b, sum([test_array[x, i] for i, x in enumerate(b)]))