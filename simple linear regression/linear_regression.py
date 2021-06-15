import numpy as np
import math
import matplotlib.pyplot as plt

class training_models:
    '''
    Choose a specific training model here
    :the selected training model should be use as the first argument to the linear_model.train_model()
    '''

    @staticmethod
    def error_check(X,Y):
        if not type(X) is np.ndarray:
            raise TypeError('X needs to be an object of array type eg.: [numpy.ndarray, list, tuple etc.]')
        if not type(X) is np.ndarray:
            raise TypeError('Y needs to be an object of array type eg.: [numpy.ndarray, list, tuple etc.]')
        if len(X)!=len(Y):
            raise ValueError('Arguments X & Y needs to have the same number of elements')

    @staticmethod
    def lrtm_simple(X,Y):

        '''
        Linear Regression Model Training Function
        : Simple Linear Regression
        @argument : X = x-axis values, Y = y-axis values
        @return   : np.array([slope,intercept])
        '''
        
        training_models.error_check(X,Y)

        N = len(X)
        XY_sum = np.sum(X*Y)
        X_sum = np.sum(X)
        Y_sum = np.sum(Y)
        X_square_sum = np.sum(np.square(X))

        slope_upper_term = (N*XY_sum)-(X_sum*Y_sum)
        slope_lower_term = (N*X_square_sum)-(X_sum**2)
        slope = slope_upper_term/slope_lower_term

        intercept_upper_term = (X_sum*XY_sum)-(Y_sum*X_square_sum)
        intercept_lower_term = (X_sum**2)-(N*X_square_sum)
        intercept = intercept_upper_term/intercept_lower_term

        return np.array([slope,intercept])

    @staticmethod
    def lrtm_pearson(X,Y):

        '''
        Linear Regression Model Training Function
        : uses 'Pearson Correlation Coefficient' to find the slope
        @argument : X = x-axis values, Y = y-axis values
        @return   : np.array([slope,intercept])
        '''
        
        training_models.error_check(X,Y)

        n = len(X)

        x_mean = np.mean(X)
        y_mean = np.mean(Y)

        xi_xm = X-x_mean
        yi_ym = Y-y_mean

        xi_xm_square = np.square(xi_xm)
        yi_ym_square = np.square(yi_ym)

        xi_xm_square_sum = np.sum(xi_xm_square)
        yi_ym_square_sum = np.sum(yi_ym_square)

        r_upper_term = np.sum(xi_xm*yi_ym)
        r_lower_term = math.sqrt(xi_xm_square_sum*yi_ym_square_sum)
        r = r_upper_term/r_lower_term

        x_deviation = math.sqrt(xi_xm_square_sum)/(n-1)
        y_deviation = math.sqrt(yi_ym_square_sum)/(n-1)

        slope = r*(y_deviation/x_deviation)
        intercept = y_mean - (slope*x_mean)

        return np.array([slope,intercept,r])

class linear_model:

    def __init__(self, slope=1, intercept=0, relation_coeficient=1):
        '''instatiates a training model object '''
        self.__slope = slope
        self.__intercept = intercept
        self.__relation_coeficient = relation_coeficient

    def train_model(self,training_function,x_data,y_data):
        '''
        Finds the best-fit line of linear model using the given datasets
        @argument1 : training_model.xxxxxxx - this is any method in the training_model class
        @argument2 : x datasets - a list, or np.ndarray
        @argument3 : y datasets - a list, or np.ndarray
        '''
        pair = training_function(x_data,y_data)
        self.__slope = pair[0]
        self.__intercept = pair[1]
        self.__relation_coeficient = pair[2]

    def regress_single(self,x_input):
        '''
        @argument : one scalar x value, int of float
        @returns : b0 + (b1*x), b0 is also called the intercept, and b1 is the slope
        : by default, if a linear model is not trained, the coeficients b0 & b1 will be set to
        : b0 = 0, b1 = 1
        '''
        return self.__intercept+(self.__slope*x_input)

    def regress_multiple(self,X_INPUTS):
        '''
        @argument : takes an array of x values
        @returns : an array of y value = b0 + (b1*x_i), b0 is also called the intercept, and b1 is the slope
        : by default, if a linear model is not trained, the coeficients b0 & b1 will be set to
        : b0 = 0, b1 = 1
        '''
        outputs = []
        for i in x:
            outputs.append(self.regress_single(i))
        return np.array(outputs)

    def get_slope_intercept(self):
        '''@returns : tuple(slope,intercept,r)'''
        return (self.__slope,self.__intercept,self.__relation_coeficient)


if __name__=='__main__':

    # datasets to be feed to the linear model
    x_year = np.array([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013])
    y_inch = np.array([40,39,41,29,32,30,35,15,10,11,20,24,10,15])

    # generates an x input values just for the sake of ploting the line later
    x = np.array(range(x_year[0]-5,x_year[len(x_year)-1]+5))

    # instantiate a linear model
    persons_model = linear_model()

    # choose a training model, feed it with the x and y datasets, then train/find the best-fit line
    persons_model.train_model(training_models.lrtm_pearson,x_year,y_inch)

    # the line of code below will generate y values from an array of x inputs, the y values are generated from the trained linear model
    # it will use the calculate slope and intercept after training the linear model (y = a + bx) to find x values
    pearsons_line = persons_model.regress_multiple(x)

    # plot the points of the original dataset
    plt.plot(x_year,y_inch,'o',color='g')

    # plot best-fit line
    plt.plot(x,pearsons_line,label='Pearson\s Corelation Coeficient Line',color='r')

    # design plot
    plt.title('Year Inch')
    plt.xlabel('year')
    plt.ylabel('amount inches')
    plt.legend(loc=0)
    # plt.xlim(-5,35)
    # plt.ylim(-5,35)
    plt.show()

    # display linear models status
    slope1, intercept1, r = persons_model.get_slope_intercept()
    print("Slope     = ",slope1)
    print("Intercept = ",intercept1)
    print("R^2       = ",(r**2))