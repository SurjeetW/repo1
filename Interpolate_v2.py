import numpy as np
import pandas as pd


def importFEM():
    dfnode = pd.read_csv('Case1_Nodes.csv', header=None)
    xy = dfnode.iloc[:,0:3]
    xy = xy.to_numpy(dtype=np.float32)
    # print('coord', xy)
    dfelement = pd.read_csv('Case1_Elements.csv', header=None)
    elements = dfelement.iloc[:,:]
    elements = elements.to_numpy(dtype=np.float32)
    # print('elem' , elements)
    dfpoint = pd.read_csv('Prediction.csv', header=None)
    points = dfpoint.iloc[:, 1:3]
    points = points.to_numpy(dtype=np.float32)
    print('point' , points)
    return xy, elements, points

def importTemp():
    dftemp = pd.read_csv('Nodal_temp_1.csv', header=None)
    temp = dftemp.iloc[:,:]
    temp = temp.to_numpy(dtype=np.float32)
    return temp


def locatepoint(point, element, xy):
    a = xy[int(element[1]-1),1:]
    b = xy[int(element[2]-1),1:]
    c = xy[int(element[3]-1),1:]
    denom = (b[1] - c[1])*(a[0] - c[0]) + (c[0] - b[0])*(a[1] - c[1])
    alpha = ((b[1] - c[1])*(point[0] - c[0]) + (c[0] - b[0])*(point[1] - c[1])) / denom
    beta = ((c[1] - a[1])*(point[0] - c[0]) + (a[0] - c[0])*(point[1] - c[1])) / denom
    gamma = 1.0 - alpha - beta
    if alpha >= 0 and beta >= 0 and gamma >= 0:
        return a, b, c, point, element
    else:
        return None
    
def interpolate_field_valueT(point, a, b, c, temp, element):
    x, y = point[0], point[1]
    x1,y1 = a[0], a[1]
    x2,y2 = b[0], b[1]
    x3,y3 = c[0], c[1]
    node1_field_value = temp[int(element[1]-1),1:]
    node2_field_value = temp[int(element[2]-1),1:]
    node3_field_value = temp[int(element[3]-1),1:]
    detA = abs(0.5*(1*(x2*y3 - x3*y2) -x1*(y3 - y2) + y1*(x3 - x2)))
    detA1 = abs(0.5*(1*(x2*y3 - x3*y2) -x*(y3 - y2) + y*(x3 - x2)))
    detA2 = abs(0.5*(1*(x3*y1 - x1*y3) -x*(y1 - y3) + y*(x1 - x3)))
    N1 = detA1 / detA
    N2 = detA2 / detA
    N3 = 1 - N1 - N2
    field_value = N1 * node1_field_value + N2 * node2_field_value + N3 * node3_field_value
    return field_value


def main():
    # fname = 'Nodal_temp_'
    n_f = 120
    data = list(range(1,n_f+1))
    print(data)
    for cntr_1 in data:
        dftemp = pd.read_csv('Nodal_temp_%d' %(cntr_1) + '.csv', header=None)
        temp = dftemp.iloc[1:,:]
        temp = temp.to_numpy(dtype=np.float32)
        
        output = importFEM()
        if output is None:
            print("Error: importFEM() returned None")
            return
        xy = output[0]
        elements = output[1]
        points = output[2]
    
        abc_list = []
        field_value_list = []
    
        for p in points:
            for e in elements:
                outputI = locatepoint(p, e, xy)
                if outputI is None:
                    pass
                else:
                    a = outputI[0]
                    b = outputI[1]
                    c = outputI[2]
                    point = outputI[3]
                    element = outputI[4]
                    abc = [a, b, c]
                    abc_list.append(abc)
                    field_value = interpolate_field_valueT(point, a, b, c, temp, element)
                    field_value_list.append(field_value)
                    print("list=", abc_list)
                    print("FV=", field_value_list)
                    break
            CSToFile = [str(tuple(x[0])) + ',' + str(tuple(x[1])) + ',' + str(tuple(x[2])) for x in abc_list]
            with open('AdjPoints.txt', "w") as f:
                for line in CSToFile:
                    f.write(line + "\n")
            with open('FV_%d' %(cntr_1) + '.txt', 'w') as f:
                for item in field_value_list:
                    f.write(str(item) + '\n')

    # dfadjpoints = pd.read_csv('AdjPoints.txt', header=None)
    # dfadjpoints[[0,2,4]] = dfadjpoints[[0,2,4]].apply(lambda x: x.str.replace('(', '', regex=False))
    # dfadjpoints[[1,3,5]] = dfadjpoints[[1,3,5]].apply(lambda x: x.str.replace(')', '', regex=False))
    # dfadjpoints.to_numpy(dtype=np.float32)

    # PredTemp = np.zeros((len(abc_list),1))
    # np.append(PredTemp, field_value, axis=1)
    # np.savetxt('PredTemp.txt',PredTemp)


if __name__ == '__main__':
    main()
