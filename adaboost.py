import numpy as np


def loadSimpData():
    datMat = np.matrix([[1, 2.1],
                        [2, 1.1],
                        [1.3, 1],
                        [1, 1],
                        [2, 1]])
    classLabels = [1, 1, -1, -1, 1]
    return datMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((dataMatrix.shape[0], 1))
    if threshIneg == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArray, classLabels, D):  # D is Weight
    dataMatrix = np.mat(dataArray)
    labelMat = np.mat(classLabels).T
    m, n = shape(dataMatrix)
    numsteps = 10.0;
    bestStump = {};
    bestClassEst = dataMatrix[:, i].max();
    minError = np.inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min();
        rangeMax = dataMatrix[L, i].max();
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stupClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                print "split: dim %d, thresh %.2f, hresh ineqal: %s, the weighted error is %.3f" % (
                    i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst


def addBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = dataArr.shape[0]
    D = np.mat(ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEST = buildStump(dataArr, classLabels, D)
        print "D:", D.T
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.apend(bestStump)
        print "classEst: ", classEST.T
        expon = np.multiply(-1 * alpha * np.map(classLabels).T, classEst)
        D = np.multiply(D, exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEST
        print "aggClassEst: ", aggClassEst.T
        aggErrors = np.multiply(sign(aggClassEst) != np.mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print "total error: ", errorRate, "\n"
        if errorRate == 0.0: break
    return weakClassArr
dataMat,classLabels = loadSimpData()
classifierArray = addBoostTrainDS(dataMat,classLabels,9)
