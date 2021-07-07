// cpp version of MLR.py
#include <iostream>
#include <cyfre/cyfre.hpp>

template<typename T>
class linear_regression
{
    private:

        cyfre::mat<T> weights;

    public:

        T regress(cyfre::mat<T> x)
        {
            long double y_hat = weights(0,0);
            for(size_t i=0; i<x.width; ++i)
            {
                y_hat+=weights(0,i+1)*x(0,i);
            }
            return y_hat;
        }

        void train(const cyfre::mat<T>& X, const cyfre::mat<T>& Y)
        {
            cyfre::mat<T> inner = cyfre::transpose(X)*X;
            inner.inv();
            cyfre::mat<T> inverse_mul_XT = inner*cyfre::transpose(X);
            cyfre::mat<T> b_hat = inverse_mul_XT*cyfre::transpose(Y);
            weights = cyfre::transpose(b_hat);
        }

        void status()
        {
            std::cout<<"Beta Coefficients : ";
            cyfre::display(weights);
        }

};

int main()
{
    linear_regression<long double> test1;
    
    cyfre::mat<long double> Y({40,39,41,29,32,30,35,15,10,11,20,24,10,15});

    std::vector<long double> x1 = {2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013};
    std::vector<long double> x0_ones(x1.size(),1);

    cyfre::mat<long double> X({x0_ones,x1});
    X.transpose();

    std::cout<<"X = ";
    cyfre::display(X);
    std::cout<<"\nY = ";
    cyfre::display(Y);

    test1.train(X,Y);
    test1.status();

    std::vector<long double> xtest = {2015};
    std::cout<<"Test : "<<test1.regress(xtest)<<'\n';

    return 0;
}

/* to compile this file you need to download the matrix library "cyfre"
   that I created, it is uploaded in github

- if you have an internet connection, g++ compiler, and git you can use
  the command below in your terminal.

command:

git clone https://github.com/mrdcvlsc/cyfre.git
g++ -o MLR.o MLR.cpp -I ./

*/
