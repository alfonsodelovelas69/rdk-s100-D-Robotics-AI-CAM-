# RDK S100 / Orange Pi - Plate Recognition

Цей проект призначений для розпізнавання українських номерних знаків на автомобілях з двох відеопотоків, їхнього порівняння з базою номерів у файлі блокнота та формування короткого електричного імпульсу на одному з контактів PIO плати RDK S100 D-Robotics при збігу.

> Увага: скрипт призначений для статичних камер і безпечного застосування в мирних цілях.

## Що реалізовано

- одночасне читання двох RTSP/HTTP джерел;
- виявлення автомобілів на S100 BPU через YOLOE11;
- пошук номерного знака в нижній частині авто;
- OCR з перевіркою формату AA1234BB;
- порівняння номерів з /root/plates_notebook.txt;
- запуск одного імпульсу на PIO при збігу;
- резервне збереження кропів номерів і логів.

---

## 1. Підготовка файлу блокнота

Через SSH/PuTTY файл створюється автоматично, якщо його ще немає. Якщо хочете зробити це вручну через термінал:

```bash
sudo mkdir -p /root
sudo nano /root/plates_notebook.txt
```

Додайте номери у форматі `AA1234BB` один на рядок, наприклад:

```text
AA1234BB
BC5678HX
AK9901OP
```

Також можна додати групово через термінал:

```bash
printf '%s\n' 'AA1234BB' 'BC5678HX' 'AK9901OP' | sudo tee -a /root/plates_notebook.txt
```

Або додати один номер:

```bash
echo 'AA1234BB' | sudo tee -a /root/plates_notebook.txt
```

Якщо файла не існує, скрипт сам створить шаблон і запише в нього приклади.

---

## 2. Як змінювати параметри запуску

Основна команда:

```bash
python3 /root/main.py \
  --stream \
  rtsp://user:pass@192.168.50.17:8554/cam1 \
  rtsp://user:pass@192.168.50.17:8554/cam2 \
  --rtsp-tcp \
  --frame-skip 2 \
  --ocr-interval 8 \
  --confirmations 2 \
  --save-unknown \
  --pio-pin 17 \
  --pio-duration 0.35
```

Пояснення параметрів:

- `--stream` — одне або два RTSP/HTTP джерела;
- `rtsp://user:pass@192.168.50.17:8554/cam1` — перша камера;
- `rtsp://user:pass@192.168.50.17:8554/cam2` — друга камера;
- `--rtsp-tcp` — використовувати TCP для RTSP;
- `--frame-skip 2` — обробляти кожен другий кадр;
- `--ocr-interval 8` — повторна OCR перевірка через 8 кадрів;
- `--confirmations 2` — підтвердження 2 рази перед збереженням;
- `--save-unknown` — зберігати невідомі номери;
- `--pio-pin 17` — контакт PIO, який активується при збігу;
- `--pio-duration 0.35` — тривалість імпульсу в секундах.

Щоб змінити адресу або параметри, просто замініть значення в команді.

---

## 2. Команда запуску для двох RTSP-камер

Для Raspberry/Orange Pi або локального хоста з доступом до двох потоків:

```bash
python3 /workspace/main.py \
  --stream \
  rtsp://user:pass@192.168.50.17:8554/cam1 \
  rtsp://user:pass@192.168.50.17:8554/cam2 \
  --rtsp-tcp \
  --frame-skip 2 \
  --ocr-interval 8 \
  --confirmations 2 \
  --save-unknown \
  --copy-desktop \
  --pio-pin 17 \
  --pio-duration 0.35
```

Якщо потрібен запуск одного каналу:

```bash
python3 /workspace/main.py \
  --stream "rtsp://user:pass@192.168.50.17:8554/cam1" \
  --rtsp-tcp \
  --frame-skip 2 \
  --ocr-interval 8 \
  --confirmations 2 \
  --save-unknown \
  --pio-pin 17 \
  --pio-duration 0.35
```

---

## 3. Команда запуску для HTTP/MJPEG

```bash
python3 /workspace/main.py \
  --stream "http://SERVER:PORT/video" \
  --frame-skip 2 \
  --ocr-interval 8 \
  --confirmations 2 \
  --save-unknown \
  --pio-pin 17 \
  --pio-duration 0.35
```

---

## 4. Orange Pi Zero 3W

### 4.1. Локальний RTSP на Orange Pi

Якщо потоки транслюються на самому Orange Pi:

```text
rtsp://user:pass@127.0.0.1:8554/cam1
rtsp://user:pass@127.0.0.1:8554/cam2
```

Через Wi‑Fi в локальній мережі:

```text
rtsp://user:pass@192.168.50.17:8554/cam1
rtsp://user:pass@192.168.50.17:8554/cam2
```

### 4.2. Безпечний віддалений доступ через SSH-тунель

На комп'ютері:

```bash
ssh -L 8554:127.0.0.1:8554 orangepi@192.168.50.17
```

Після цього у VLC відкрийте:

```text
rtsp://user:pass@127.0.0.1:8554/cam1
```

---

## 5. RDK S100 D-Robotics

На платі RDK S100 потрібно мати:

- готове BPU-середовище;
- модель YOLOE11 для S100;
- доступ до /opt/hobot/model/...;
- Python-середовище з OpenCV, pytesseract, rapidfuzz.

Запуск на платі:

```bash
python3 /root/main.py \
  --stream \
  rtsp://user:pass@192.168.50.17:8554/cam1 \
  rtsp://user:pass@192.168.50.17:8554/cam2 \
  --rtsp-tcp \
  --frame-skip 2 \
  --ocr-interval 8 \
  --confirmations 2 \
  --save-unknown \
  --pio-pin 17 \
  --pio-duration 0.35
```

---

## 6. Збіжність і PIO

Коли OCR визначає номер і він збігається з записом у /root/plates_notebook.txt або дуже близький до нього, скрипт:

- записує кроп номерного знака в /root/crops;
- якщо увімкнено --copy-desktop, копіює його у /home/sunrise/Desktop/plate_crops;
- генерує короткий імпульс на PIO pin, за замовчуванням 17.

У логах побачите рядки на кшталт:

```text
[MATCH] source=... plate=AA1234BB votes=2 trigger=PIO17
```

---

## 7. Корисні параметри

```bash
--stream           один або кілька URL-адрес RTSP/HTTP
--rtsp-tcp         TCP для RTSP
--frame-skip       крок обходу кадрів
--ocr-interval     інтервал OCR перевірки для одного track
--confirmations    кількість підтверджень до запису
--save-unknown     зберігати невідомі номери
--copy-desktop     копіювати кропи в робочий стіл
--pio-pin          номер GPIO/PIO контакту
--pio-duration     тривалість імпульсу в секундах
```

---

## 8. Файли проекту

- [main.py](main.py) — основний скрипт обробки
- [tests/test_plate_match_gpio.py](tests/test_plate_match_gpio.py) — базова перевірка логіки збігу та PIO
- [README.md](README.md) — інструкція запуску

---

## 9. Відомі обмеження

- проект розрахований на статичні камери в безпечному середовищі;
- для рухомих або динамічних об’єктів потрібні додаткові зміни в алгоритмі відстеження;
- для реального запуску на платі необхідно перевіряти GPIO/PIO та доступ до камер саме в конкретному обладнанні.





Коротка команда для копіювання в PuTTY


python3 /root/main.py --stream rtsp://user:pass@192.168.50.17:8554/cam1,rtsp://user:pass@192.168.50.17:8554/cam2 --rtsp-tcp --frame-skip 2 --ocr-interval 8 --confirmations 2 --save-unknown --pio-pin 17 --pio-duration 0.35



Для одного каналу

python3 /root/main.py --stream "rtsp://user:pass@192.168.50.17:8554/cam1" --rtsp-tcp --frame-skip 2 --ocr-interval 8 --confirmations 2 --save-unknown --pio-pin 17 --pio-duration 0.35



Для HTTP

python3 /root/main.py --stream "http://SERVER:PORT/video" --frame-skip 2 --ocr-interval 8 --confirmations 2 --save-unknown --pio-pin 17 --pio-duration 0.35


