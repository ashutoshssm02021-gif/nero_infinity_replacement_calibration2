
def find_calibration_points2(a, date_times):
    calibration_points = []
    irrigation_detected=False

    # for i in range(1, len(a), 1):
    i = 0
    while i < len(a):
        if a[i] > 0:
            irrigation_detected=True
        if a[i] > 0 and (a[i]>0.01) and irrigation_detected:
            valid_irr_point = i
            min_point = None
            while a[i] >= 0 and i < len(a) - 1:
                i+=1
            if a[i] <= 0:
                min_point=i
                while a[i] <= 0 and i < len(a) - 1:
                    # if a[i] < -0.003 :
                    #     break
                    if a[i] < a[min_point]:
                        if not a[i] < 1.4*a[min_point]:
                            break
                        min_point = i
                    i+=1
                i = min_point + 1
                if a[min_point] < 0:
                    while i < len(a) - 1  and a[i] < -0.006 :
                        i += 1      
                    if i < len(a) and a[i] >= - 0.006 and a[i] <= 0:
                        calibration_points.append(date_times[i])
                        irrigation_detected=False
                    else:
                        i = valid_irr_point+1
                else:
                    i = min_point

        i += 1
    return calibration_points


if __name__ == '__main__':
    # a = [0, 0.016, 0, -0.0207, -0.0203, -0.00221]
    # indices = list(range(len(a)))
    # a = [-0.00222, 0.0108, -0.0043, 0, -0.00222]
    # indices = list(range(len(a)))
    # a = [0.01611, 0, -0.02067, -0.02034, -0.00221]
    # indices = list(range(len(a)))
    # a=[0.0235, 0, 0.01002, 0, 0, 0, -0.00262, -0.01161, -0.00787, -0.00783, -0.0021]
    # a=[0.01611, 0, -0.02067, -0.02034, -0.00221]
    # ans = -0.00221
    # a=[0.04311, -0.01148, 0.01611, 0, -0.02067, -0.02034, -0.00221]
    # ans = -0.00221
    # a=[0, 0, 0.04311, -0.01148, 0.01611, 0, -0.02067, -0.02034, -0.00221]
    # ans = -0.00221
    # a=[0.04141, 0.0111, 0, 0, 0.04311, -0.01148, 0.01611, 0, -0.02067, -0.02034, -0.00221]
    # ans = -0.00221
    # a=[0.00863, 0, 0, -0.00201, 0, 0, 0.00865, 0.04141, 0.0111, 0, 0, 0.04311, -0.01148, 0.01611, 0, -0.02067, -0.02034, -0.00221]
    # ans = -0.00221
    a=[0.02519, 0, -0.00151, 0.04433, 0, 0, 0, 0, -0.00943, -0.00938, -0.01406, -0.00624, -0.00303]
    ans = -0.00303
    indices = list(range(len(a)))
    cal_indices = find_calibration_points2(a, indices)
    if cal_indices:
        print(a[cal_indices[0]] == ans)
        print(a[cal_indices[0]])