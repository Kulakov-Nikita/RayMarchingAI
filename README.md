### Ray Marching AI

---

Проект по исследованию возможности апроксимации, с помощью глубокого машинного обучения, функции Space Distance Field (SDF), используемой в алгоритме рендеринга изображений Ray Marching.

Для работы необходим Python 3.13

Установка зависимостей:
```bash
pip install -r requirements.txt
```

Для запуска обучения модели:
```bash
python deepSDF.py
```

Для рендеринга изображения:
```bash
python render_to_file.py <output_path> <width> <height> --max_step <max_step> --precision <precision> --max_dist <max_dist>
```
* output_path - куда сохранить изображение
* width, height - длина и ширина изображения в пикселях
  
Опиционально:
* max_step - максимальное количество шагов Ray Marching
* precision - доверительный интервал в Ray Marching
* max_dist - дальность прорисовки

Пример:
```bash
python render_to_file.py frame.jpg 1280 720 --max_step 30 --precision 0.1 --max_dist 10
```
![](https://github.com/Kulakov-Nikita/RayMarchingAI/blob/main/frame.jpg)
