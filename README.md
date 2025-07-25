# Red Light Violation Detection System🚦
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-orange)
![DeepSORT](https://img.shields.io/badge/DeepSORT-Tracking-lightgrey)

![Violation Example](web_interface.png)  
*Vehicle crosses violation zone while the light is red. System logs the clip and details.*

## Table of Contents
- [Motivation](#motivation)
- [Introduction](#introduction)
- [Features](#features)
- [Installation & Usage](#installation--usage)
- [Contributing](#contributing)
- [Acknowledgents](#acknowledgements)
- [Author](#author)
- [Contact](#contact)

## Motivation
This project was built during my internship to solve a real-world problem: Detecting Red Light Traffic Violations.\
It uses computer vision to automate the monitoring process and aims to improve road safety by reducing manual surveillance efforts

## Introduction
Urban traffic violations like red-light jumping are a growing concern, leading to accidents and safety risks. Manual enforcement alone is not enough. This system uses YOLOv8 and DeepSORT to detect red light violations in real-time and helps authorities enforce rules more effectively. It features a web interface that logs violations with clips and timestamps, making monitoring both efficient and user-friendly.
## Features  
- **Vehicle Detection:** Identifies vehicles (cars, bikes, trucks, buses, etc.) using YOLOv8 trained on the BDD100K dataset.  
- **Traffic Light Classification:** Detects and classifies traffic light states (Red, Yellow, Green) using YOLOv8 trained on the BSTLD dataset.  
- **Consistent Vehicle Tracking:** Uses DeepSORT to assign persistent IDs to vehicles across frames.  
- **Polygon-Based Violation Zones:** Enables flexible zone shapes for more accurate violation detection compared to basic rectangular ROIs.  
- **Automatic Violation Capture:** Records short MP4 clips of vehicles crossing during red light and saves them as evidence.  
- **CSV Violation Logs:** Stores violation details — Timestamp, Vehicle ID, vehicle Class, and Video Clip in organized logs.  
- **Web Interface:** Flask-based UI supports video upload and live camera feeds for easy usage.  
- **Clean Output Management:** Organizes all logs and video evidence neatly in output folders for review.

### Installation & Usage
**1. Clone the repository:**
```bash
git clone https://github.com/webserver105/Red_Light_Violation_Detection
cd red_light_violation_detection
```
**2. Install dependencies:**
```bash
pip install -r requirements.txt
git clone https://github.com/ZQPei/deep_sort_pytorch
# Extract this library in Red_Light_Violation_Detection folder only
```
**3. Usage:**
```bash
python app.py
```
- The default polygon coordinates in the code are set up for the sample video `demo4.mp4`.  
- If you're using a different video, **you'll need to manually set new coordinates** that match your camera angle or scene.

> 🎯 **How to get the coordinates?**
> 
> Here's my super high-tech method: **MS Paint** 🖌️😄  
> Just pause your video on a key frame, take a screenshot, open it in MS Paint, and note down the (x, y) positions by hovering over the desired points.  
> Then update those coordinates in the code! 💻🎯

## Contributing
This project is open source so others can easily get involved. If you'd like to contribute, please fork the repository, create a feature branch, and open a pull request. All kinds of contributions bug fixes, features, or suggestions — are welcome!

## Acknowledgements
This project uses the following open datasets. Huge thanks to the authors and contributors for making these resources publicly available:

- **[BDD100K](https://bdd-data.berkeley.edu/)**  
  *Yu, Fisher, et al. "BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning."*  
  *IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020.*

- **[BSTLD (Bosch Small Traffic Lights Dataset)](https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset)**  
  *Behrendt, Karsten, and Novak, Libor. "A Deep Learning Approach to Traffic Lights: Detection, Tracking, and Classification."*  
  *IEEE International Conference on Robotics and Automation (ICRA), 2017.*

<details>
<summary><strong>BibTeX for BDD100K</strong></summary>

```bibtex
@InProceedings{bdd100k,
    author = {Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
    title = {BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
```

</details>

<details>
<summary><strong>BibTeX for BSTLD</strong></summary>
  
```bibtex
@inproceedings{BehrendtNovak2017ICRA,
  title={A Deep Learning Approach to Traffic Lights: Detection, Tracking, and Classification},
  author={Behrendt, Karsten and Novak, Libor},
  booktitle={Robotics and Automation (ICRA), 2017 IEEE International Conference on},
  organization={IEEE}
}
```
</details>

## Author
Kunal Gandvane, kunal7sr@gmail.com\
Student at Department of Civil Engineering\
Indian Institute of Technology Bombay

## Contact
For any inquiries or further information, please contact me at [LinkedIn.](https://www.linkedin.com/in/kunal-gandvane-b28062346/)
