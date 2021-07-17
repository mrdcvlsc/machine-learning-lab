import numpy as np

class KaimingHe:
    '''ideal to use with layers that uses ReLu as an activaton'''

    @staticmethod
    def normal(input_weight_count=0,output_weight_count=0):
        return 0, np.sqrt(2/input_weight_count)

    @staticmethod
    def uniform(input_weight_count=0,output_weight_count=0):
        bound = np.sqrt(6/input_weight_count)
        return -bound,bound

class Xavier:
    '''ideal to use with layers that uses sigmoid as an activation'''

    @staticmethod
    def normal(input_weight_count=0,output_weight_count=0):
        return 0, np.sqrt(2/(input_weight_count+output_weight_count))

    @staticmethod
    def uniform(input_weight_count=0,output_weight_count=0):
        bound = np.sqrt(6)/np.sqrt(input_weight_count+output_weight_count)
        return -bound,bound

class basicInit:
    '''just a plain sad initialization'''

    @staticmethod
    def uniform(input_weight_count=0,output_weight_count=0):
        bound = 1/np.sqrt(input_weight_count)
        return -bound,bound
