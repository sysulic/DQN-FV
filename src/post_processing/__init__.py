import post_processing.post_precessing as post_precessing
import post_processing.greedy as greedy

def predict_without_post_processing(data, is_test=False):
    imprecise_result = greedy.imprecise(data, is_test)
    precise_result = greedy.precise(data, is_test)
    return imprecise_result, precise_result

def predict_with_post_processing(data, alphas, is_test=False):
    #imprecise_result = post_precessing.imprecise(data, alphas['imprecise'], is_test)
    #precise_result = post_precessing.precise(data, alphas['precise'], is_test)
    precise_result = post_precessing.precise(data, alphas, is_test)
    #return imprecise_result, precise_result
    return precise_result
