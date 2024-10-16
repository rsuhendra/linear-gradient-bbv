from BBV_optimize import *

inputVals = np.array([0.725, -0.56, 0.5, 0.39, 0.65, 0.05, 0.75, 0.067])

bbv = BBV_gradient_taxis_point(weights=inputVals, p1=0.75, p2=0, brait=True, mode_motor='ou', mode_taxis='derivative')

eval_obj = Evaluate_gradient(bbv)

test = eval_obj.evaluate()
