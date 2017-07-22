## conditional variational autencoder for keras [(link to my post about it)](http://nnormandin.com/science/2017/07/01/cvae.html)

This is an implementation of a CVAE in Keras trained on the MNIST data set, based on the paper [Learning Structured Output Representation using Deep Conditional Generative Models ](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=0ahUKEwiamruIr5HVAhXFVj4KHRh0BG4QFggxMAE&url=https%3A%2F%2Fpdfs.semanticscholar.org%2F3f25%2Fe17eb717e5894e0404ea634451332f85d287.pdf&usg=AFQjCNGP9YZk7oDH-pyk_2_V3dAPJEiMbg) and the code fragments from Agustinus Kristiadi's blog [here](http://wiseodd.github.io/techblog/2016/12/17/conditional-vae/).

Since a one-hot vector of digit class labels is concatenated with the input prior to encoding and again to the latent space prior to decoding, we can sample individual digits and combinations. This gif shows a transition along the number line from zero to nine in increments of approximately 0.033. 

![MNIST_CVAE](http://i.makeagif.com/media/7-17-2017/qWHlvW.gif)
