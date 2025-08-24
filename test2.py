
def find_calibration_points2(a, date_times):
    calibration_points = []
    irrigation_detected=False

    # for i in range(1, len(a), 1):
    i = 0
    while i < len(a):
        if a[i] > 0:
            irrigation_detected=True
        if a[i] > 0 and (a[i]>0.01) and irrigation_detected:
            min_point = None
            while a[i] > 0 and i < len(a) - 1:
                i+=1
            if a[i] <= 0:
                min_point=i
                while a[i] <= 0 and i < len(a) - 1:
                    if a[i] < a[min_point]:
                        min_point = i
                    i+=1
                i = min_point + 1
                while a[i] < -0.006 and i < len(a) - 1 :
                    i += 1      
                if a[i] >= - 0.006 and a[i] < 0:
                    calibration_points.append(date_times[i])
                    irrigation_detected=False

        i += 1
    return calibration_points


if __name__ == '__main__':
    # a = [0, 0.016, 0, -0.0207, -0.0203, -0.00221]
    # indices = [0, 1, 2, 3, 4, 5]
    a = [-0.00222, 0.0108, -0.0043, 0, -0.00222]
    indices = [0, 1, 2, 3, 4]
    print(find_calibration_points2(a, indices))