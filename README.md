<h1 align="center">🎱 billiard-analyzer</h1>

<p align="center">
  <b>Курсовой проект: Анализ динамики и кинематики игры в бильярд</b><br>
  <i>Математика, физика и визуализация на грани игры и науки</i>
</p>

---

## 🚀 О проекте

`billiard-analyzer` — это инструмент для анализа движения бильярдных шаров с применением методов обработки сигналов и визуализации.  
Проект охватывает:

- 📈 Фильтрацию координат с помощью **фильтра Калмана**
- 🎯 Обнаружение и классификацию **столкновений**
- 🔮 Прогнозирование будущего движения
- 🧩 Экспорт в `UNIGINE` для 3D-визуализации

---

## 🧠 Технологии

- `Python 3.10+`
- `NumPy`, `Matplotlib`, `JSON`
- `UNIGINE` (C#/C++) — для 3D-анимации
- ✨ Математическое моделирование движения

---

---

## 🗂️ Структура проекта

Проект организован так, чтобы удобно было работать с кодом, данными, визуализациями и документацией.  
Вот как устроено всё под капотом:

| Путь              | Назначение                                                                 |
|-------------------|---------------------------------------------------------------------------|
| `src/`            | 🧠 Основной код проекта — логика фильтрации, предсказания, событий         |
| └── `main.py`     | 🎯 Главный скрипт: от загрузки данных до генерации результата              |
| `requirements.txt`| 📌 Зависимости проекта для быстрого запуска через `pip install -r`        |
| `README.md`       | 📖 Этот самый файл — красиво рассказывает о проекте                        |
| `.gitignore`      | 🚫 Исключения для Git — чтобы лишние файлы не попадали в репозиторий       |

