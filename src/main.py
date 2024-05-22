from convexity_1comp import generate_data
from plotter import generate_plots
from value_individual import generate_plots_individual
from comp_time_individual import generate_plots_comp
from best_fit_params import generate_best_fit
from meta_tree import generate_plots_maximum_beliefs, generate_plots_maximum_expansions

if __name__=="__main__":
    for k in [4,8,12]:
        generate_data(k)
    generate_plots()
    generate_plots_individual()
    generate_plots_comp()
    generate_best_fit()
    
    #Here we show that our solutions are robust to the approximation schemes
    
    #This is for k_c
    generate_plots_maximum_expansions([1,2,3])		
    #This is for k, the values to put in are k/2. Your PC may run into memory issues. Use values more than 7 carefully (i.e. k>14).
    generate_plots_maximum_beliefs([3,4,5])
