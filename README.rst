.. image:: /docs/source/img/golem_logo-02.png
   :alt: Logo of GOLEM framework
   :align: center
   :width: 500

.. class:: center

    |python| |pypi| |build| |docs| |license| |tg| |eng|


Оптимизация и обучение графовых моделей эволюционными методами
--------------------------------------------------------------

GOLEM - это фреймворк искусственного интеллекта с открытым исходным кодом для оптимизации и изучения структурированных
моделей на основе графов с помощью метаэвристических методов. Он основан на двух идеях:

1. Метаэвристические методы имеют большой потенциал в решении сложных задач.

Фокус на метаэвристике позволяет работать с типами задач, для которых градиентные методы обучения (в частности, нейронные сети)
не могут быть легко применены. Например для задач многоцелевой оптимизации или для комбинаторных задач.

2. Структурированные модели важны в различных областях.

Обучение на основе графов позволяет находить решения в виде структурированных и гибридных вероятностных моделей, не говоря
уже о том, что широкий спектр задач в разных предметных областях естественным образом формулируется в виде графов.

В совокупности это представляет собой подход к ИИ, который потенциально приводит к созданию структурированных, интуитивно понятных,
поддающихся интерпретации методов и решений для широкого круга задач.


Основные возможности
====================

- **Структурированные модели** с одновременной оптимизацией структуры графа и его свойств (атрибутов узлов).
- **Метаэвристические методы** (в основном эволюционные), применимые к любой задаче с четко заданной целевой функцией.
- **Многоцелевая оптимизация**, которая может учитывать как качество, так и сложность.
- **Оптимизация с ограничениями** с поддержкой произвольных ограничений, специфичных для конкретных областей.
- **Расширяемость** для новых предметных областей.
- **Интерпретируемость** благодаря метаэвристике, структурированным моделям и инструментам визуализации.
- **Воспроизводимость** благодаря подробной истории оптимизации и сериализации моделей.


Применение
==========

GOLEM потенциально применим к любой структуре задач оптимизации:

- к задачам, которые могут быть представлены в виде направленных графов;
- к задачам, которые имеют какую-то четко определенную фитнес-функцию.

Графовые модели могут представлять собой фиксированные структуры (например, физические модели, такие как ферменные конструкции)
или функциональные модели, которые определяют поток данных или процесс предсказания (например, байесовские сети, которые
могут быть обучены и могут отвечать на запросы).

Примеры применения GOLEM:

- Автоматическое машинное обучение (AutoML) для поиска оптимальных пайплайнов машинного обучения в `фреймворке FEDOT <https://github.com/aimclub/FEDOT>`_
- Поиск структуры при помощи байесовской сети в `фреймворке BAMT <https://github.com/ITMO-NSS-team/BAMT>`_
- Поиск дифференциальных уравнений для физических моделей в рамках `фреймворка EPDE <https://github.com/ITMO-NSS-team/EPDE>`_
- Геометрический дизайн физических объектов в рамках `фреймворка GEFEST <https://github.com/aimclub/GEFEST>`_
- `Поиск архитектуры нейронных сетей <https://github.com/ITMO-NSS-team/nas-fedot>`_

Поскольку GOLEM - это фреймворк общего назначения, легко представить его потенциальное применение, например,
поиск конечных автоматов для управления в робототехнике или изучение молекулярных графов для разработки лекарств и
многое другое.


Установка
=========

GOLEM можно установить с помощью ``pip``:

.. code-block::

  $ pip install thegolem


Структура проекта
=================

Репозиторий включает в себя следующие пакеты и папки:

- Пакет ``core`` содержит основные классы и скрипты.
- Пакет ``core.adapter`` отвечает за преобразование между графами из предметной области и внутренним представлением, используемым оптимизаторами.
- Пакет ``core.dag`` содержит классы и алгоритмы для изображения и обработки графов.
- Пакет ``core.optimisers`` содержит оптимизаторы для графов и все вспомогательные классы (например, те, которые представляют фитнес, отдельных лиц, популяции и т.д.), включая историю оптимизации.
- Пакет ``core.optimisers.genetic`` содержит генетический (также называемый эволюционным) оптимизатор графов и операторы (мутация, отбор и так далее).
- Пакет ``core.utilities`` содержит утилиты и структуры данных, используемые другими модулями.
- Пакет ``serializers`` содержит класс ``Serializer`` и отвечает за сериализацию классов проекта (графики, история оптимизации и все, что с этим связано).
- Пакет ``visualisation`` содержит классы, которые позволяют визуализировать историю оптимизации, графы и некоторые графики, полезные для анализа.
- Пакет ``examples`` включает в себя несколько примеров использования фреймворка.
- Все модульные и интеграционные тесты содержатся в каталоге ``test``.
- Источники документации находятся в каталоге ``docs``.


Текущие исследования/разработки и планы на будущее
==================================================

Наша научно-исследовательская команда открыта для сотрудничества с другими научными коллективами, а также с партнерами из индустрии.

Как участвовать
===============

- Инструкция для добавления изменений находится в `репозитории </docs/source/contribution.rst>`__.

Благодарности
=============

Мы благодарны контрибьютерам за их важный вклад, а участникам многочисленных конференций и семинаров -
за их ценные советы и предложения.

Разработка ведётся при поддержке
================================

.. image:: /docs/source/img/AIM-Strong_Sign_Norm-01_Colors.svg
    :width: 400px
    :align: center
    :alt: Strong AI in industry logo

Разработка поддерживается исследовательским центром `Сильный искусственный интеллект в промышленности <https://sai.itmo.ru/>`__ `Университета ИТМО <https://itmo.ru/>`__.

Контакты
========
- `Telegram канал <https://t.me/FEDOT_helpdesk>`_ для решения проблем и ответов на вопросы, связанные с FEDOT
- `Команда Лаборатории моделирования естественных систем <https://itmo-nss-team.github.io/>`_
- `Анна Калюжная <https://scholar.google.com/citations?user=bjiILqcAAAAJ&hl=ru>`_, тимлид (anna.kalyuzhnaya@itmo.ru)
- `Новости <https://t.me/NSS_group>`_
- `Youtube канал <https://www.youtube.com/channel/UC4K9QWaEUpT_p3R4FeDp5jA>`_

Цитирование
===========

Если вы используете наш проект в своей работе или исследовании, мы будем признательны за цитирование.

@article{nikitin2021automated,
  title = {Automated evolutionary approach for the design of composite machine learning pipelines},
  author = {Nikolay O. Nikitin and Pavel Vychuzhanin and Mikhail Sarafanov and Iana S. Polonskaia and Ilia Revin and Irina V. Barabanova and Gleb Maximov and Anna V. Kalyuzhnaya and Alexander Boukhanovsky},
  journal = {Future Generation Computer Systems},
  year = {2021},
  issn = {0167-739X},
  doi = {https://doi.org/10.1016/j.future.2021.08.022}}

@inproceedings{polonskaia2021multi,
  title={Multi-Objective Evolutionary Design of Composite Data-Driven Models},
  author={Polonskaia, Iana S. and Nikitin, Nikolay O. and Revin, Ilia and Vychuzhanin, Pavel and Kalyuzhnaya, Anna V.},
  booktitle={2021 IEEE Congress on Evolutionary Computation (CEC)},
  year={2021},
  pages={926-933},
  doi={10.1109/CEC45853.2021.9504773}}


Другие статьи можно найти на `ResearchGate <https://www.researchgate.net/project/Evolutionary-multi-modal-AutoML-with-FEDOT-framework>`_.

.. |docs| image:: https://readthedocs.org/projects/thegolem/badge/?version=latest
    :target: https://thegolem.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |build| image:: https://github.com/aimclub/GOLEM/actions/workflows/unit-build.yml/badge.svg?branch=main
   :alt: Build Status
   :target: https://github.com/aimclub/GOLEM/actions/workflows/unit-build.yml

.. |coverage| image:: https://codecov.io/gh/aimclub/GOLEM/branch/main/graph/badge.svg
   :alt: Coverage Status
   :target: https://codecov.io/gh/aimclub/GOLEM

.. |pypi| image:: https://img.shields.io/pypi/v/thegolem.svg
   :alt: PyPI Package Version
   :target: https://img.shields.io/pypi/v/thegolem

.. |python| image:: https://img.shields.io/pypi/pyversions/thegolem.svg
   :alt: Supported Python Versions
   :target: https://img.shields.io/pypi/pyversions/thegolem

.. |license| image:: https://img.shields.io/github/license/aimclub/GOLEM
   :alt: Supported Python Versions
   :target: https://github.com/aimclub/GOLEM/blob/main/LICENSE.md

.. |downloads_stats| image:: https://static.pepy.tech/personalized-badge/thegolem?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads
   :target: https://pepy.tech/project/thegolem

.. |tg| image:: https://img.shields.io/badge/Telegram-Group-blue.svg
   :alt: Telegram Chat
   :target: https://t.me/FEDOT_helpdesk

.. |by-golem| image:: http://img.shields.io/badge/powered%20by-GOLEM-orange.svg?style=flat
   :target: http://github.com/aimclub/GOLEM
   :alt: Powered by GOLEM

.. |eng| image:: https://img.shields.io/badge/lang-en-red.svg
            :target: /README_en.rst