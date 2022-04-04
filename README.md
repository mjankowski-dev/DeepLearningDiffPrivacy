
<h2>1. Introduction:</h2>
Recent explosive progress of the machine learning and deep learning industry raises questions about the privacy guarantess of this field. Due to its nature, training neural networks requires large and extensive datasets, which are mostly crowdsourced. They can often contain sensitive information which is why this topic is important . This blogpost is about reproducing the fidings of the paper "Deep learning with Diferential privacy" by M. Abadi, A. Chu et al. Specificaly the diferential privacy mechanisms are combined with machine learning methods to train neural networks within single digit privacy budget. The next section gives insight into Differential privacy, which is followed by implementation of DP mechanisms. Section 4 presents the experimental setup of the network. Results and findings are presented in section 5.
<h2>2. Differential privacy</h2>
Differential privacy can be seen as a standard for privacy guarantee for algorithms that rely on data-bases. An algorithm is called differentialy private if by looking at the output, one cannot tell if certain individual data was included or not. This means that the behavior of the algorithm doesn't change whether the inputs from one individual are or are not in the dataset. Consider two adjacent data bases, which means that they differ with at least a single entry. The output of the algorithm based on either of the databases shouldn't differ more than epsilon, to satisfy the delta,epsilon differential privacy.
<p><img src="https://miro.medium.com/max/700/1*IKhOQqMSkinUSGpD16XYpw.png" alt="Differential privacy" width="700" height="379" /></p>

Mathematical definition of differential privacy is as follows: An algorithm K gives ε-differential privacy if for  data sets D and D′, differing by at most one row, and any S (input) ⊆ Range(K) 
<p><img style="display: block; margin-left: auto; margin-right: auto;" src="https://i.ibb.co/s5Sz7xH/diffpr.png" alt="epsilon, delta differential privacy" width="439" height="60" /></p>

Epsilon is a metric of privacy loss, the lower the more privacy is guaranteed. Delta is an additional metric of probability of breaking the plain epsilon differential privacy.

Differential privacy deals with this problem by adding "noise"or randomness to the data, which prevents identifying any individual data points. Instead of returning the raw data, the algorithm return an approximation of the data. Intuitively, the noise level is related to the accuracy of the algorithm, the more noise we introduce, the higher the privacy, but the accuracy will be lower.
<h2>3. Differential privacy mechanisms:</h2>
<h3><span style="color: #99ccff; background-color: #ffffff;"><strong>2.1 DP-SGD</strong></span></h3>
<h3><span style="color: #99ccff; background-color: #ffffff;"><strong>2.2 Moments accountant</strong></span></h3>
<h3><span style="color: #99ccff; background-color: #ffffff;"><strong>2.3 DP-PCA</strong></span></h3>
<h2>4. Experimental setup:</h2>
<h3><span style="color: #99ccff; background-color: #ffffff;"><strong>3.1 MNIST</strong></span></h3>
<h3><span style="color: #99ccff; background-color: #ffffff;"><strong>3.2 Preprocessing</strong></span></h3>
<h3><span style="color: #99ccff; background-color: #ffffff;"><strong>3.3 Architecture</strong></span></h3>
<h3><span style="color: #99ccff; background-color: #ffffff;"><strong>3.4 Effects of parameters</strong></span></h3>
<h2>5. Results:</h2>
<p>&nbsp;</p>
<h2>5. Conclusion:</h2>
