## Implementation of Contrastive Language-Image Pre-Training (CLIP) in TensorFlow for Medical Imaging

This project implements CLIP from OpenAI [^1] in TensorFlow using the ROCO (Radiology Objects in COntext) dataset [^2]. CLIP aligns image and text representations through contrastive learning, enabling zero-shot learning.

The ROCO dataset, containing radiology images and captions, was used to train the CLIP model specifically for medical applications.

For more details, refer to the [Project Report](https://github.com/Calonca/clip-tensorflow/blob/main/CLIP_Tensorflow_report.pdf).

## References
[^1]: CLIP Code (OpenAI). Available at: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)

[^2]: O. Pelka, S. Koitka, J. Rückert, F. Nensa, C.M. Friedrich,  
["__Radiology Objects in COntext (ROCO): A Multimodal Image Dataset__"](https://labels.tue-image.nl/wp-content/uploads/2018/09/AM-04.pdf).  
MICCAI Workshop on Large-scale Annotation of Biomedical Data and Expert Label Synthesis (LABELS) 2018, September 16, 2018, Granada, Spain. Lecture Notes on Computer Science (LNCS), vol. 11043, pp. 180-189, Springer Cham, 2018.  
doi: [10.1007/978-3-030-01364-6_20](https://doi.org/10.1007/978-3-030-01364-6_20)

