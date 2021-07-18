# Transportation-AI
* This project is developed as a part of qualifying test given by Dr. Xiang Liu, Rutgers University.
* Task:
  * Use pretrained model like Yolov3, Yolov5 and perform object detection. <br>
  * Develop a web-app to showcase the output and identified objects.
* Submission:
  * In the web-app you can upload single/ multiple images and server will perform object detection and will return the image with bounding boxes.
  * You can also see the detailed statastics - about objects and their frequencies in the given image.
  * **Output:** 
       * Output of the given images can be found in [here](https://github.com/vraj152/Transportation-AI/tree/master/static/OutputImages)
       * Video of the output can be found [here](https://github.com/vraj152/Transportation-AI/tree/master/output)
* Tech stack used:
  * Backend: Flask
    * Pretrained model used: [Yolov5](https://github.com/ultralytics/yolov5)
  * Frontend: HTML, JavaScript, CSS, AJAX
* Code summary:
  * *Note: code is explained in detail in comments.
  * Backend: 
    * There are 2 endpoints.
      1. [/ (GET)](https://github.com/vraj152/Transportation-AI/blob/1ee85a68aac6bfde123d20ef3856eae54475d8a0/flaskAPI.py#L20) : Which renders the home page where you can select the image(s).
      2. [/objectDetect (POST)](https://github.com/vraj152/Transportation-AI/blob/1ee85a68aac6bfde123d20ef3856eae54475d8a0/flaskAPI.py#L24) : Which will accept the list of images and perform object detection and will return appropriate response.
    * flaskAPI.py consists of these 2 endpoints and also contains multiple helper functions.
  * Frontend:
    * home.html : There is only one HTML file required. Since AJAX is used, response will be rendered on the same page without refreshing the page.
* Credits:
  * Swiper -> [GitHub repo](https://github.com/nolimits4web/Swiper)
  * Modal (to display stats) -> [Codepen](https://codepen.io/edznan/pen/JMJQbr)
* Project Credits: [Dr. Xiang Liu & team (Rutgers CAIT)](https://cait.rutgers.edu/directory/xiang-liu/)
