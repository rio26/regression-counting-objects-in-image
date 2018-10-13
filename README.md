## Running regression task in order to count the number of people in a picture

### Task 1
<p> Train 5 regression models.
  
  * least-squares (LS)
  * regularized LS (RLS)
  * L1-regularized LS (LASSO)
  * robust regression (RR)
  * Bayesian regression (BR)

<p> Training Data for task 1:
  
  * sampx - sample input values (xi), each entry is an input value.
  * sampy - sample output values (yi), each entry is an output.
  * polyx - input values for the true function (also these are the test inputs x_*).
  * polyy - output values for the true function.
  * thtrue - the true value of the \theta used to generate the data.
  
### Task 2
<p> Estimate the number of people in the picture
  
  * trainx -- training inputs. Each column is a 9-dimensional feature vector.
  * trainy -- training outputs (mean-subtracted counts for each input vector).
  * testx -- test inputs.
  * testy -- true outputs for the test inputs.
