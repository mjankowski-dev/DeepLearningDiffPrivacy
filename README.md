
<h2>1. Introduction:</h2>
Recent explosive progress of the machine learning and deep learning industry raises questions about the privacy guarantess of this field. Due to its nature, training neural networks requires large and extensive datasets, which are mostly crowdsourced. They can often contain sensitive information which is why this topic is important . This blogpost is about reproducing the fidings of the paper "Deep learning with Diferential privacy" by M. Abadi, A. Chu et al. Specificaly the diferential privacy mechanisms are combined with machine learning methods to train neural networks within single digit privacy budget. The next section gives insight into Differential privacy, which is followed by implementation of DP mechanisms. Section 4 presents the experimental setup of the network. Results and findings are presented in section 5.
<h2>2. Differential privacy</h2>
Differential privacy can be seen as a standard for privacy guarantee for algorithms that rely on data-bases. An algorithm is called differentialy private if by looking at the output, one cannot tell if certain individual data was included or not. This means that the behavior of the algorithm doesn't change whether the inputs from one individual are or are not in the dataset. Consider two adjacent data bases, which means that they differ with at least a single entry. The output of the algorithm based on either of the databases shouldn't differ more than epsilon, to satisfy the delta,epsilon differential privacy.
<p><img src="https://miro.medium.com/max/700/1*IKhOQqMSkinUSGpD16XYpw.png" alt="Differential privacy" width="700" height="379" /></p>
<p style="text-align: center;"><em>Source: <a href="https://privacytools.seas.harvard.edu/files/privacytools/files/pedagogical-document-dp_new.pdf">Kobbi Nissim</a></em></p>


Mathematical definition of differential privacy is as follows: An algorithm K gives ε-differential privacy if for  data sets D and D′, differing by at most one row, and any S (input) ⊆ Range(K) 
<p><img style="display: block; margin-left: auto; margin-right: auto;" src="https://i.ibb.co/s5Sz7xH/diffpr.png" alt="epsilon, delta differential privacy" width="439" height="60" /></p>

Epsilon is a metric of privacy loss, the lower the more privacy is guaranteed. Delta is an additional metric of probability of breaking the plain epsilon differential privacy.

Differential privacy deals with this problem by adding "noise"or randomness to the data, which prevents identifying any individual data points. Instead of returning the raw data, the algorithm return an approximation of the data. Intuitively, the noise level is related to the accuracy of the algorithm, the more noise we introduce, the higher the privacy, but the accuracy will be lower.
<h2>3. Differential privacy mechanisms:</h2>
<h3><span style="color: #99ccff; background-color: #ffffff;"><strong>2.1 DP-SGD</strong></span></h3>
The optimizer plays a vital role in the differential privacy framework. Through a series of steps it attempts to mask the incoming gradients in order to prevent them from becoming too informative with regard to respective samples. Furthermore, following a different interpretation this operation can also potentially fight overfitting to some degree. \\
The optimizing scheme works through four main steps. To start, the gradient based on the loss of every sample is obtained for every layer in the network using  the jacobian on a batch of samples. Once this is obtained this gradient is clipped, which is applied for each layer separately. This means that the gradients are normalized by the l2-norm scaled by a clipping constant C. Notice that clipping only occurs whenever l2-norm > C, otherwise the gradients pass through unmodified. After this all gradients per sample are combined into a single set to which noise is added. This noisy gradient is then used to update the parameters. These steps are included in their mathematical notation:  
<p><img style="display: block; margin-left: auto; margin-right: auto;" src="https://iili.io/MwyHxt.jpg" alt="Pseudo code" width="491" height="254" /></p>

<h3><span style="color: #99ccff; background-color: #ffffff;"><strong>2.2 Moments accountant</strong></span></h3>
<p>The moment accountant keeps track of the pivacy loss encountered. Every time the optimization step is applied a small loss privacy in privacy is experienced. This loss can be computed and represented through either &epsilon or &delta. The main factor governing the magnitude of this loss is the noise level applied in the optimizer (&sigma)<p>. If this is known and epsilon is kept fixed (as in our experiments) the corresponding delta can be computed. The moments for the loss variable are obtained by an operation between the probability density functions of a Gaussian and a Gaussian mixture distribution. In practice, the values for E1 and E2 are approximated over a large sample space of 10k over the 2 distributions.

<p><img style="display: block; margin-left: auto; margin-right: auto;" src="https://iili.io/MwpybI.jpg" alt="Moment1" width="418" height="258" /></p>

This moment is computed for every possible lambda under consideration. For every iteration the respective moments are continuously summed using the composability theorem:

<p><img style="display: block; margin-left: auto; margin-right: auto;" src="https://iili.io/MwppON.jpg" alt="Moment2" width="216" height="85" /></p>

After this has been done the most limiting moment can be identified. Note that this might change between iterations, experical evidence shows a tendency to select higher moments first, yet this selection scales down to lower moments as the iterations continue. Using this moment the delta can be computed and checked against a threshold. As long as this threshold is not exceeded training continues. The corresponding delta can be obtained according to: 
<p><img style="display: block; margin-left: auto; margin-right: auto;" src="https://iili.io/MwpmRp.jpg" alt="Moment3" width="282" height="84" /></p>

<h3><span style="color: #99ccff; background-color: #ffffff;"><strong>2.3 DP-PCA</strong></span></h3>
Principal component analysis is a strong tool used to approximate high dimensional data in fewer dimensions. Low dimensional approximation still preserves most of the information, at a smaller variable amount. This is beneficial for the  
<h2>4. Reproduced setup:</h2>
<h3><span style="color: #99ccff; background-color: #ffffff;"><strong>3.1 MNIST</strong></span></h3>
The data set used to reproduce the findings of the paper was MNIST dataset which is a 
<h3><span style="color: #99ccff; background-color: #ffffff;"><strong>3.2 Preprocessing</strong></span></h3>
<h3><span style="color: #99ccff; background-color: #ffffff;"><strong>3.3 Architecture</strong></span></h3>
<h3><span style="color: #99ccff; background-color: #ffffff;"><strong>3.4 Effects of parameters</strong></span></h3>
<h2>5. Results:</h2>
<p>&nbsp;</p>
<h2>5. Conclusion:</h2>
