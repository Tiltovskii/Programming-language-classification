# Programming-language-classification
Classification for 100 programming language and extensions.
------------------------------------
Это репозиторий, в котором решалась задача для [контеста от телеграмма](https://t.me/contest_ru/39), где нужно было сделать классификатор 100 языков программирования и разрешений, такких как `.csv`, `.xml`, `.json` и др.

Датасет
------------------------------------
Датасет был собран из [каггловского](https://www.kaggle.com/datasets/joonasyoon/file-format-detection/code), [rosetta с hf](https://huggingface.co/datasets/cakiki/rosetta-code), диалоги на русском [SiberianPersonaChat](https://huggingface.co/datasets/SiberiaSoft/SiberianPersonaChat) и остальное все чего не хватало или было мало, а это непопулярные языки, парсились с GitHub.

Модели
------------------------------------
Модели были протестированы как классические: `RandomForest`, `LogReg`, `Gradient Boosting`, `Fasttext`, так и глубинные: `RNN`, [CodeBert](https://huggingface.co/huggingface/CodeBERTa-language-id), который в основе использует RoBerta.

По качеству хуже всего показали себя `RandomForest`, `LogReg`, `Fasttext` и `CodeBert`. Последний не удался, так как было мало времени на обучение, и просто было не успеть натренировать модель + есть ограничение на модель: инциализация + предикт не должен был превышать 10мс, поэтому модель недообучилась и была брошена, но запустить её трейн можно. Лучше всего себя показали `Gradient Boosting`, который был в виде библиотеки `CatBoost`, и `RNN`, который был в виде `LSTM`. 
