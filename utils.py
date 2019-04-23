# COMMON FUNCTION
def extractdata(classification, testfile):
    """ extrac the trainning data end test data given the path
    to these files"""
    train  = pd.read_table(classification, header=None, engine='python')
    test = pd.read_table(testfile, header=None, engine='python')
    X_train, X_test = train.as_matrix()[:,0:2], test.as_matrix()[:,0:2]
    y_train, y_test = train.as_matrix()[:,2], test.as_matrix()[:,2]
    return X_train, y_train, X_test, y_test
