.. image:: /docs/source/img/golem_logo-02.png
   :alt: Logo of GOLEM framework
   :align: center
   :width: 500

.. class:: center

    |sai| |itmo|

    |python| |pypi| |build| |integration| |coverage| |docs| |license| |tg| |eng| |mirror|


Оптимизация и обучение графовых моделей эволюционными методами
--------------------------------------------------------------

GOLEM - это фреймворк искусственного интеллекта с открытым исходным кодом для оптимизации и обучения структурированных
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
- Поиск структуры при помощи байесовской сети в `фреймворке BAMT <https://github.com/aimclub/BAMT>`_
- Поиск дифференциальных уравнений для физических моделей в рамках `фреймворка EPDE <https://github.com/ITMO-NSS-team/EPDE>`_
- Геометрический дизайн физических объектов в рамках `фреймворка GEFEST <https://github.com/aimclub/GEFEST>`_
- `Поиск архитектуры нейронных сетей <https://github.com/ITMO-NSS-team/nas-fedot>`_

Поскольку GOLEM - это фреймворк общего назначения, легко представить его потенциальное применение, например,
поиск конечных автоматов для алгоритмов контроля в робототехнике или оптимизация молекулярных графов для разработки лекарств и
многое другое.


Установка
=========

GOLEM можно установить с помощью ``pip``:

.. code-block::

  $ pip install thegolem


Быстрый старт
=============

Следующий пример показывает поиск графа по графу-эталону с помощью метрики расстояния редактирования (Edit Distance). Оптимизатор настраивается с минимальным набором параметров и простыми одноточечными мутациями. Более подробные примеры можно найти в файлах `simple_run.py <https://github.com/aimclub/GOLEM/blob/main/examples/synthetic_graph_evolution/simple_run.py>`_, `graph_search.py <https://github.com/aimclub/GOLEM/blob/main/examples/synthetic_graph_evolution/graph_search.py>`_ и `tree_search.py <https://github.com/aimclub/GOLEM/blob/main/examples/synthetic_graph_evolution/tree_search.py>`_ в директории `examples/synthetic_graph_evolution <https://github.com/aimclub/GOLEM/tree/main/examples/synthetic_graph_evolution>`_.

.. code-block:: python

    def run_graph_search(size=16, timeout=8):
        # Генерируем целевой граф и целевую функцию в виде edit distance
        node_types = ('a', 'b')  # Available node types that can appear in graphs
        target_graph = generate_labeled_graph('tree', size, node_types)
        objective = Objective(partial(tree_edit_dist, target_graph))
        initial_population = [generate_labeled_graph('tree', 5, node_types) for _ in range(10)]

        # Укажем параметры оптимизации
        requirements = GraphRequirements(timeout=timedelta(minutes=timeout))
        gen_params = GraphGenerationParams(adapter=BaseNetworkxAdapter(), available_node_types=node_types)
        algo_params = GPAlgorithmParameters(pop_size=30)

        # Инициализируем оптимизатор и запустим оптимизацию
        optimiser = EvoGraphOptimizer(objective, initial_population, requirements, gen_params, algo_params)
        found_graphs = optimiser.optimise(objective)

        # Визуализируем итоговый граф и график сходимости
        found_graph = gen_params.adapter.restore(found_graphs[0])  # Transform back to NetworkX graph
        draw_graphs_subplots(target_graph, found_graph, titles=['Target Graph', 'Found Graph'])
        optimiser.history.show.fitness_line()
        return found_graph


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

Поддержка
=========

Исследование проводится при поддержке `Исследовательского центра сильного искусственного интеллекта в промышленности <https://sai.itmo.ru/>`_
`Университета ИТМО <https://itmo.ru/>`_ в рамках мероприятия программы центра: Разработка и испытания 
экспериментального образца библиотеки алгоритмов сильного ИИ в части базовых алгоритмов автоматического МО 
для структурного обучения композитных моделей ИИ, включая автоматизацию отбора признаков

Контакты
========
- `Telegram канал <https://t.me/FEDOT_helpdesk>`_ для решения проблем и ответов на вопросы, связанные с FEDOT
- `Команда Лаборатории моделирования природных систем <https://itmo-nss-team.github.io/>`_
- `Николай Никитин <https://scholar.google.com/citations?user=eQBTGccAAAAJ&hl=ru>`_, руководитель направления AutoML (nnikitin@itmo.ru)
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

Публикации, описывающие применение GOLEM для прикладных задач:
==============================================================

В данных публикациях описывается применение алгоритмов GOLEM и основанных на нем решений
для различных прикладных задач.

- Алгоритмы поиска оптимального пайплайна машинного обучения для прогнозирования временных рядов: Sarafanov M., Pokrovskii V., Nikitin N. O. Evolutionary Automated Machine Learning for Multi-Scale Decomposition and Forecasting of Sensor Time Series //2022 IEEE Congress on Evolutionary Computation (CEC). – IEEE, 2022. – С. 01-08.

- Алгоритмы идентификации структуры уравнения для акустических данных: Hvatov A. Data-Driven Approach for the Floquet Propagator Inverse Problem Solution //ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). – IEEE, 2022. – С. 3813-3817.

- Алгоритмы идентификации структуры дифференциальных уравнений в частных производных: Maslyaev M., Hvatov A. Solver-Based Fitness Function for the Data-Driven Evolutionary Discovery of Partial Differential Equations //2022 IEEE Congress on Evolutionary Computation (CEC). – IEEE, 2022. – С. 1-8.

- Алгоритмы структурного обучения сетей: Deeva I., Kalyuzhnaya A. V., Alexander V. Boukhanovsky Adaptive Learning Algorithm for Bayesian Networks Based on Kernel Mixtures Distributions//International Journal of Artificial Intelligence. – 2023. - Т.21. - №. 1. - С. 90.

.. |docs| image:: https://readthedocs.org/projects/thegolem/badge/?version=latest
    :target: https://thegolem.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |build| image:: https://github.com/aimclub/GOLEM/actions/workflows/unit-build.yml/badge.svg?branch=main
   :alt: Build Status
   :target: https://github.com/aimclub/GOLEM/actions/workflows/unit-build.yml

.. |integration| image:: https://github.com/aimclub/GOLEM/actions/workflows/integration-build.yml/badge.svg?branch=main
   :alt: Integration Build Status
   :target: https://github.com/aimclub/GOLEM/actions/workflows/integration-build.yml

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

.. |ITMO| image:: https://github.com/aimclub/open-source-ops/blob/add_badge/badges/ITMO_badge_rus.svg
   :alt: Acknowledgement to ITMO
   :target: https://itmo.ru

.. |SAI| image:: https://github.com/aimclub/open-source-ops/blob/add_badge/badges/SAI_badge.svg
   :alt: Acknowledgement to SAI
   :target: https://sai.itmo.ru/

.. |mirror| image:: https://camo.githubusercontent.com/9bd7b8c5b418f1364e72110a83629772729b29e8f3393b6c86bff237a6b784f6/68747470733a2f2f62616467656e2e6e65742f62616467652f6769746c61622f6d6972726f722f6f72616e67653f69636f6e3d6769746c6162
   :alt: GitLab mirror for this repository
   :target: https://gitlab.actcognitive.org/itmo-nss-team/GOLEM