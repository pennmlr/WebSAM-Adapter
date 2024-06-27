# WebSAM-Adapter
## Alok Shah, Roberto Ligeralde, Gaurav Goel, Paul Loh
An implementation of [WebSAM-Adapter](https://link.springer.com/chapter/10.1007/978-3-031-56027-9_27) from Ren et al.

# Model Architecture
![model_architecture](https://media.springernature.com/full/springer-static/image/chp%3A10.1007%2F978-3-031-56027-9_27/MediaObjects/551041_1_En_27_Fig2_HTML.png?as=webp)
The adapter consists of 3 parts: Edge-Component Tuning, Patch-Embedding Tuning, and *k adapter units. The EC and PE Tunes are learned linear layers applied to the Sobel-Filtered input image and the encoded patches respectively, with the *i-th adapter unit using an MLP to feed the results to the *i-th block of SAM's image encoder.

# Training
We train the model on [Webis-WebSeg-20](https://webis.de/data/webis-webseg-20.html), a dataset of 8,490 web page images with 42,450 ground truth segmentations.
