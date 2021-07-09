# LifelongMixtureVAEs
The implementation of Lifelong Mixture of Variational Autoencoders


Title : Lifelong Mixture of Variational Autoencoders

# Paper link


# Abstract

 In this paper, we propose an end-to-end lifelong learning mixture of experts. Each expert is implemented by a Variational Autoencoder (VAE). The experts in the mixture system are jointly trained by maximizing a mixture of individual component evidence lower bounds (MELBO) on the log-likelihood of the given training samples. The mixing coefﬁcients in the mixture model, control the contributions of each expert in the global representation. These are sampled from a Dirichlet distribution whose parameters are determined through non-parametric estimation during the lifelong learning. The model can learn new tasks fast when these are similar to those previously learnt. The proposed Lifelong mixture of VAE (L-MVAE) expands its architecture with new components when learning a completely new task.
 After the training, our model can automatically determine the relevant expert to be used when fed with new data samples. This mechanism benefits both the memory efficiency and the required computational cost as only one expert is used during the inference.  The L-MVAE inference model is able to perform interpolations in the joint latent space across the data domains associated with different tasks and is shown to be efficient for disentangled learning representation. 


# Environment

1. Tensorflow 1.5
2. Python 3.6

# BibTeX

