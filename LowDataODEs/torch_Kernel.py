import torch
import numpy as np
import gpytorch
from torch.optim import Adam

# Each Kernel class carrys a exact gp model
class Kernel(object):
    def __init__(self, train_x, train_y, theta=None, sigma=None, nugget=1e-4):
        """
        Parameters
        ----------
        theta:  vector of shape n_hyperparameters
                vector containing the hyperparameters of this kernel
                theta[0] is the multiplicative constant
                theta[1] is the lengthscale
        sigma:  scalar
                std of the observation noise
        nugget: scalar
                small float that is added to the diagonal of the kernel matrix
                CPhi to guarantee positive definiteness also numerically.
        """
        self.train_x = train_x
        self.train_y = train_y
        self.theta = theta
        self.sigma = sigma
        self.nugget = nugget

        self.model = self._build_gp_model()

    def _build_gp_model(self):
        '''Override in children classes'''
        raise NotImplementedError

    def getNTheta():
        """
        Returns the amount of parameters needed by this kernel in the theta
        vector
        """
        raise NotImplementedError(
            "getNTheta has not been implemented for this kernel.")

    def setHyperparams(self, theta, sigma, newNugget=False):
        """
        Sets the hyperparameters of this kernel
        """
        raise NotImplementedError

    def getBounds(self, y, time):
        """
        creates the bounds for the optimization of the hyperparameters.
        
        Parameters
        ----------
        y:          vector
                    observation of the states. Target of the regression
        time:       vector
                    time points of the observations. Input of the regression
        Returns
        ----------
        bounds: list of theta.size + 1 pairs of the form 
                (lowerBound, upperBound), representing the bounds on the 
                kernel hyperparameters in theta, while the last one is the
                bound on sigma
        """
        raise NotImplementedError(
            "getBounds has not yet been implemented for this kernel!"
            )

    def learnHyperparams(self, theta0, sigma0, y, time, normalize=False,
                         standardize=False, newNugget=False, trainingIter=100, learning_rate=0.01):
        """
        Learns the hyperparameters by maximizing the marginal likelihood of the
        data y

        Parameters
        ----------
        theta0:     vector
                    initial guess for parameters for optimization
        sigma0:     scalar
                    initial guess for noise for optimization
        y:          vector of length nObs or array of shape nObs x nReps
                    observation of the states. Target of the regression
                    if y is an array, it is assumed that the observations
                    come from different, independent experiments on the same
                    time scale.
        time:       vector of length nObs
                    time points of the observations. Input of the regression
        normalize:  boolean
                    if True, hyperparameters will be optimized for the
                    mean corrected observation.
                    if False, hyperparameters will be optimized directly
        standardize:    boolean
                        if True, hyperparameters will be optimized for the
                        standardized observations. normalize will be ignored
                        if False, hyperparameters will be optimized as
                        specified by normalize keyword.
        newNugget:  False or float
                    if false, the old nugget will be used
                    if float, the old nugget will be overwritten
                    nugget is the small number that is added to the GP prior
                    covariance matrix to guarantee positive definiteness
                    also numerically
        trainingIter:  scalar
                    Training iterations
        """
        # define optimization target
        def negLogLikelihood(params):
            # for multiple trajectories, just add likelihood of each
            # run. Assumes one GP per trajectory and mean likelihood as
            # optimization target
            pass

        # set optimizer settings
        bounds = self.getBounds(y, time)
        print(bounds)
                  
        # set nugget
        if not (isinstance(newNugget, bool) and not newNugget):
            self.nugget = newNugget
        else:
            print(newNugget)
        
        ### Actually Train
        if len(self.train_y.shape) == 1:
            train_y = train_y.reshape(-1, 1)  # Ensure y is 2D
            
        train_x = torch.tensor(self.train_x, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.float32)
        
        n_outputs = train_y.shape[1]
        
        # Initialize a model and likelihood for each output dimension
        self.models = []
        self.likelihoods = []
        
        for i in range(n_outputs):
            # Create a likelihood with initial noise (sigma in original code)
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            
            # Add nugget to the likelihood noise for numerical stability
            likelihood.noise = self.nugget
            
            # Create the model
            model = self._build_gp_model()
            
            self.models.append(model)
            self.likelihoods.append(likelihood)
        
        # Set models to training mode
        for model in self.models:
            model.train()
            
        for likelihood in self.likelihoods:
            likelihood.train()
            
        # Initialize optimizer with all parameters from all models
        # This is key to ensuring shared hyperparameters across outputs
        # We'll extract and tie the kernel parameters later
        parameters = []
        for model in self.models:
            parameters.extend(list(model.parameters()))
            
        optimizer = Adam(parameters, lr=learning_rate)
        
        # Get references to kernel parameters from first model
        # These will be tied across all models
        base_outputscale = self.models[0].covar_module.outputscale
        base_lengthscale = self.models[0].covar_module.base_kernel.lengthscale
        
        # Define loss function (negative log likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood
        
        # Training loop
        for i in range(trainingIter):
            # Zero gradients
            optimizer.zero_grad()
            
            # Calculate loss (sum of negative log likelihoods across all outputs)
            loss = 0
            for j, (model, likelihood) in enumerate(zip(self.models, self.likelihoods)):
                # Tie kernel parameters across all models to ensure shared hyperparameters
                if j > 0:
                    model.covar_module.outputscale = base_outputscale
                    model.covar_module.base_kernel.lengthscale = base_lengthscale
                
                # Add this model's NLL to the total loss
                output = model(train_x)
                loss += -mll(likelihood, model)(output, train_y[:, j])
            
            # Backpropagate
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                print(f'Iteration {i+1}/{trainingIter} - Loss: {loss.item()}')
                print(f'  Outputscale: {base_outputscale.item()}')
                print(f'  Lengthscale: {base_lengthscale.item()}')
                # In GPyTorch, likelihood.noise_covar.noise corresponds to sigma^2 in original code
                print(f'  Noise: {self.likelihoods[0].noise_covar.noise.item()}')
        
        # Set models to evaluation mode
        for model in self.models:
            model.eval()
            
        for likelihood in self.likelihoods:
            likelihood.eval()
            
        self.is_fitted = True
        
        # Return fitted hyperparameter values
        return {
            'outputscale': base_outputscale.item(),  # Corresponds to theta[0] in original code
            'lengthscale': base_lengthscale.item(),  # Corresponds to theta[1] in original code
            'sigma': np.sqrt(self.likelihoods[0].noise_covar.noise.item())  # Convert variance to std
        }
                                    
        print("Kernel optimization output: ")

        # check for positive semidefinite
        # C = self.getCPhi(time)
        # minEigenvalue = np.min(np.linalg.eig(C)[0])
        # print("minimum eigenvalue = {}".format(minEigenvalue))
        # if (minEigenvalue < 1e-5):
        #     print("\n\nRECOMMENDATION: USE BIGGER NUGGET\n\n")
        # C = C + (self.sigma**2)*np.eye(C.shape[0])
        # try:
        #     # test for psd
        #     np.linalg.cholesky(C)
        # except:
        #     print("matrix not positive semidefinite")

    def k(self, time1, time2):
        """
        returns the correlation between time1 and time2 for the specific kernel
        this does not yet add the observation noise
        """
        raise NotImplementedError("k has not been implemented for this kernel")
    
    def CDash(self, time1, time2):
        """
        returns the derivative of the correlation between time1 and time2 with
        respect to time2, used in the C_Phi' matrix
        """
        raise NotImplementedError(
            "CDash has not been implemented for this kernel")
    
    def DashC(self, time1, time2):
        """
        returns the derivative of the correlation between time1 and time2 with
        respect to time1, used in the 'C_Phi matrix
        """
        raise NotImplementedError(
            "DashC has not been implemented for this kernel")
    
    def CDoubleDash(self, time1, time2):
        """
        returns the derivative of the correlation between time1 and time2 with
        respect to both times, used in the C_Phi'' matrix
        """
        raise NotImplementedError(
            "CDoubleDash has not been implemented for this kernel")
    
    def getCPhi(self, time):
        """
        returns the correlation matrix of the GP using this kernel
        """
        raise NotImplementedError

    def getCPhiDash(self, time):
        """
        returns the derivative of C_Phi w.r.t. the second time argument
        """
        raise NotImplementedError

    def getDashCPhi(self, time):
        """
        returns the derivative of C_Phi w.r.t. the first time argument
        """
        raise NotImplementedError

    def getCPhiDoubleDash(self, time):
        """
        returns the derivative of C_Phi w.r.t. both time arguments
        """
        raise NotImplementedError
    
        
    def predict(self, time):
        """
        """
        raise NotImplementedError