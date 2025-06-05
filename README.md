# Geospatial Models for Humanitarian Applications

This summary report compiles the current state-of-the-art GeoAI models that are open and publicly available, with a strong focus on humanitarian use cases. Each entry includes the model name, purpose, openness level (code, weights, data), framework,License, and links to access the model or repository. The list is organized thematically for easier review.

### 1. **Humanitarian Mapping & Infrastructure Models**

| **Model Name** | **Purpose** | **Openness (Weights / Code / Data)** | **Framework** | **Output** | **License** | **Source** |
| --- | --- | --- | --- | --- | --- | --- |
| **RAMP** | Building footprint segmentation for health & planning | Yes / Yes / Yes | TensorFlow | Segmentation masks | MIT | [GitHub](https://github.com/devglobalpartners/ramp-code) |
| **YOLO** | Object Detection , Optimized for building footprint | Yes/Yes/Yes | PyTorch | Segmentation masks | GNU Affero General Public License v3.0 | [GitHub](https://github.com/hotosm/fAIr-utilities/tree/master/hot_fair_utilities/model) |
| **PGRID** | Electric grid pole/line detection from drone imagery | Partial / Yes / Partial | PyTorch | GeoJSON poles + lines | MIT | [GitHub](https://github.com/USAFORUNHCRhive/turkana-grid-mapping) |
| **Solar Farming Model** | Solar Farm Area from Satellite Imagery | Partial/ Yes/ Permission Requited | PyTorch | Not Sure | MIT and CDLA-Permissive-2.0 for the data  | [GitHub](https://github.com/microsoft/solar-farms-mapping) |
| **Solar Panels and Rooftop materials Model** | From USAFORUNHCR and Microsoft for mapping solar panels and roof materials | Partial/ Yes/ Permission Requited | PyTorch | Semantic segmentation | MIT | [GitHub](https://github.com/USAFORUNHCRhive/turkana-camp-roof-mapping) |
| **Open Cities AI** | Building segmentation from drone images | Yes / Yes / Yes | PyTorch | Segmentation masks | MIT | [DrivenData GitHub](https://github.com/drivendataorg/open-cities-ai-challenge) |
| **SpaceNet Buildings (Baseline)** | Building footprint segmentation from high-res satellite | Yes / Yes / Yes | TensorFlow | Segmentation masks | Apache 2.0 | [CosmiQ GitHub](https://github.com/avanetten/CosmiQ_SN7_Baseline) |
|  |  |  |  |  |  |  |

### 2. **Disaster Response Models**

| **Model Name** | **Purpose** | **Openness (Weights / Code / Data)** | **Framework** | **Output** | **License** | **Source** |
| --- | --- | --- | --- | --- | --- | --- |
| **SKAI (WFP)** | Damage detection from pre/post disaster imagery | Yes / Yes / Partial | TensorFlow | Building-level damage classification | Apache 2.0 | [Google Research GitHub](https://github.com/google-research/skai) |
| **xView2 (1st place)** | Damage classification from satellite pre/post | Partial / Yes / Yes | PyTorch | Building + damage label | MIT | [GitHub](https://github.com/DIUx-xView/xView2_first_place) |
| **xView2 Baseline** | Segmentation + damage classification | Yes / Yes / Yes | TensorFlow | Same as Above | BSD-3 | [DIUx GitHub](https://github.com/DIUx-xView/xView2_baseline) |
| **Sen1Floods11 Baseline** | Flood detection from SAR | Reproducible / Yes / Yes |  PyTorch | Binary flood masks | MIT | [Cloud to Street GitHub](https://github.com/cloudtostreet/Sen1Floods11) |
| **SpaceNet 8** | Flood + road/building segmentation | Yes / Yes / Yes | PyTorch | Multi-class segmentation + attributes | Apache 2.0 | [GitHub](https://github.com/SpaceNetChallenge/SpaceNet8) |
| **Microsoft Building Damage Assessment** | Building damage assessments from remotely sensed imagery | NA/Yes/NA | PyTorch | Building damage percentage  | MIT | [GitHub](https://github.com/microsoft/building-damage-assessment/) |
|  |  |  |  |  |  |  |

### 3. **Agriculture & Food Security**

| **Model Name** | **Purpose** | **Openness (Weights / Code / Data)** | **Framework** | **Output** | **License** | **Source** |
| --- | --- | --- | --- | --- | --- | --- |
| **Field Boundary (NASA Harvest 1st)** | Segment smallholder field boundaries | Yes / Yes / Yes | PyTorch | Not Sure | CC 4.0 | [Radiant MLHub](https://github.com/radiantearth/model_nasa_rwanda_field_boundary_competition_gold) |
| **AgriFieldNet (1st)** | Crop classification per field | Yes / Yes / Yes | PyTorch | Field-level crop label | CC 4.0 | [Radiant MLHub](https://github.com/radiantearth/model_ecaas_agrifieldnet_gold) |
| **Spot the Crop (Winner)** | Crop classification (multi-modal) | Yes / Yes / Yes | PyTorch | Crop label | Apache 2.0 | [Radiant Earth GitHub](https://github.com/radiantearth/spot-the-crop-challenge) |
| **CV4A Kenya (ICLR)** | Crop classification with GRU/CNN | NA | NA | Crop label | Not Sure , Couldn’t Find repo | [Zindi](https://zindi.africa/competitions/iclr-workshop-challenge-2-radiant-earth-computer-vision-for-crop-recognition/data) |
| **Solar Panel Detector** | Segment solar panels in HR imagery | Yes / Yes / Yes |  TensorFlow | Binary segmentation | MIT | [GitHub](https://github.com/A-Stangeland/SolarDetection) |
|  |  |  |  |  |  |  |

### 4. **Land Use / Environmental Models**

| **Model Name** | **Purpose** | **Openness (Weights / Code / Data)** | **Framework** | **Output** | **License** | **Source** |
| --- | --- | --- | --- | --- | --- | --- |
| **Dynamic World** | 10m land cover segmentation (global) | Inference only / Code Yes / Data Yes | TensorFlow | 9-class per-pixel map | Apache 2.0 (code), CC-BY (data) | [GitHub](https://github.com/google/dynamicworld/tree/master) |
| **LandCoverNet Baseline** | Annual land cover segmentation data mainly, can be implemented via unet or transforme | Yes / Depends / Yes | Depends | Multi-class segmentation | Apache 2.0 | [Sample Implementation](https://github.com/pavlo-seimskyi/semantic-segmentation-satellite-imagery), [Data Source](https://source.coop/repositories/radiantearth/landcovernet/description)  |
| **BigEarthNet ViT / ResNet** | Multi-label land cover classifier | Yes / Yes / Yes | PyTorch | Patch-level LULC tags | MIT | [Hugging Face](https://huggingface.co/BIFOLD-BigEarthNetv2-0), [Code](https://git.tu-berlin.de/rsim/reben-training-scripts)  |
| **EuroSAT** | Land use classification | Yes / Yes / Yes | Tensorflow | Patch-level label | MIT | [GitHub](https://github.com/phelber/EuroSAT), [Collab](https://colab.research.google.com/github/e-chong/Remote-Sensing/blob/master/EuroSAT%20Land%20Cover%20Classification/EuroSAT%20Land%20Use%20and%20Land%20Cover%20Classification%20using%20Deep%20Learning.ipynb) |
|  |  |  |  |  |  |  |

### 5. **Foundation Models for EO**

| **Model Name** | **Purpose** | **Openness (Weights / Code / Data)** | **Framework** | **Output** | **License** | **Source** |
| --- | --- | --- | --- | --- | --- | --- |
| **Prithvi-EO 1.0 / 2.0** | General Earth observation embeddings | Yes / Yes / Yes | PyTorch (ViT) | EO embeddings / fine-tune | Apache 2.0 | [Hugging Face](https://huggingface.co/ibm-nasa-geospatial) |
| **Prithvi-WxC** | Climate data representation learning | Yes / Yes / Yes | PyTorch (ViT) | Climate embeddings | Apache 2.0 | [Hugging Face](https://huggingface.co/ibm-nasa-geospatial) |
| **Clay (Radiant Earth)** | EO foundation model (MAE-ViT) | Yes / Yes / Yes | PyTorch | EO embeddings | Apache 2.0 | [Hugging Face](https://huggingface.co/made-with-clay/Clay) |
| **SatMAE (MVRL)** | Self-supervised ViT for EO | Yes / Yes / Yes | PyTorch | Encoder embeddings | Attribution-NonCommercial 4.0 International | [GitHub](https://github.com/sustainlab-group/SatMAE) |
| **SAM (Meta AI)** | Segmentation-anything | Yes / Yes / Yes | PyTorch | Segmentation masks | Apache 2.0 | [GitHub](https://github.com/facebookresearch/segment-anything) |
| **GeoCLIP / SatCLIP** | Geo image-text alignment (CLIP-like) | Yes / Yes / NA | PyTorch | Text/image embeddings | MIT | [GitHub](https://github.com/VicenteVivan/geo-clip) |
| **SeCo (ResNet50)** | Seasonal contrast pretraining | Yes / Yes / Yes | PyTorch | ResNet embeddings | MIT | [GitHub](https://github.com/ServiceNow/seasonal-contrast) |
|  |  |  |  |  |  |  |

### 6. **Toolkits & Libraries**

| **Toolkit** | **Purpose** | Openness (Weights / Code / Data) | **Framework** | **Included Models** | **License** | **Source** |
| --- | --- | --- | --- | --- | --- | --- |
| **TorchGeo** | Training/evaluating EO models | Yes / Yes / Yes | PyTorch | ResNet, ViT, UNet, seasonal contrast, etc. | MIT | [GitHub](https://github.com/microsoft/torchgeo) |
| **SIMRDWN** | Multi-scale object detection (YOLO, SSD, etc.) | Yes / Yes / - | Darknet / PyTorch | YOLT, RetinaNet, SSD | MIT | [GitHub](https://github.com/CosmiQ/SIMRDWN) |
| **DeepForest** | Tree crown detection | Yes / Yes / Yes | PyTorch | Faster R-CNN | MIT | [GitHub](https://github.com/weecology/DeepForest) |
|  |  |  |  |  |  |  |

### More Sources  to checkout :

Microsoft Models : https://www.microsoft.com/en-us/research/project/geospatial-machine-learning/downloads/ 

More GeoAI Models : https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models?tab=readme-ov-file 

## Contirbution :

Feel free to edit if some information is missing or list needs to be updated ! PR's are welcome !
