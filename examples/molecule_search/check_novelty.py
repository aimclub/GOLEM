import pandas as pd

from examples.molecule_search.mol_adapter import MolAdapter
from examples.molecule_search.mol_metrics import CocrystalsMetrics, sa_score
from golem.core.optimisers.genetic.evaluation import MultiprocessingDispatcher
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory


def check_novelty_mol_path(
        train_dataset_path: str,
        gen_data: list,
        train_col_name: str,
        gen_col_name: str,
        gen_len: int ):
    """Function for count how many new molecules generated compared with train data.


    :param train_dataset_path: Path to csv train dataset.
    :param gen_data: gen molecules.
    :param train_col_name: Name of column that consist a molecule strings.
    :param gen_col_name: Name of column that consist a molecule strings.
    :return:
    """
    train_d = pd.read_csv(train_dataset_path)[train_col_name]
    gen_d = pd.DataFrame(gen_data, columns=[gen_col_name])
    duplicates = gen_d.duplicated(subset=gen_col_name, keep='first').sum()/len(gen_d)
    total_len_gen = len(gen_d[gen_col_name])
    #gen_d = gen_d[gen_d['val_check']==1][gen_col_name]
    #len_train = len(train_d)
    len_gen = len(gen_d.drop_duplicates())
    novelty =( len(gen_d[gen_col_name].drop_duplicates())-gen_d[gen_col_name].drop_duplicates().isin(train_d).sum() )/ gen_len * 100
    print('Generated molecules consist of',novelty, '% unique new examples',
          '\t',
          f'duplicates: {duplicates}')
    return novelty,duplicates


def check_novelty(
        train_dataset_path: str,
        gen_data_path: str,
        train_col_name: str,
        gen_col_name: str) ->str:
    """Function for count how many new molecules generated compared with train data.


    :param train_dataset_path: Path to csv train dataset.
    :param gen_data_path: Path to csv gen dataset.
    :param train_col_name: Name of column that consist a molecule strings.
    :param gen_col_name: Name of column that consist a molecule strings.
    :return:
    """
    train_d = pd.read_csv(train_dataset_path)[train_col_name]
    gen_d = pd.read_csv(gen_data_path)
    duplicates = gen_d.duplicated(subset=gen_col_name, keep='first').sum()
    total_len_gen = len(gen_d[gen_col_name])
    gen_d = gen_d[gen_d['val_check']==1][gen_col_name]
    #len_train = len(train_d)
    len_gen = len(gen_d)

    print('Generated molecules consist of',(len_gen-train_d.isin(gen_d).sum())/len_gen*100, '% new examples',
          '\t',f'{len_gen/total_len_gen*100}% valid molecules generated','\t',
          f'duplicates, {duplicates}')


def check_validity(exp_name_path):
    metrics = CocrystalsMetrics('CN1C2=C(C(=O)N(C1=O)C)NC=N2')
    adapter = MolAdapter()
    objective = Objective(
        quality_metrics={'orthogonal_planes': metrics.orthogonal_planes,
                         'unobstructed': metrics.unobstructed,
                         'h_bond_bridging': metrics.h_bond_bridging,
                         'sa_score': sa_score},
        is_multi_objective=True)
    evaluator = MultiprocessingDispatcher(adapter=adapter, n_jobs=-1).dispatch(objective)

    all_individuals = {'generated_coformers': [], 'unobstucted': [], 'orthogonal_planes': [], 'h_bond_bridging': [], 'sa': []}
    for i in range(10):
        print(i)
        history = OptHistory.load(fr"{exp_name_path}\history_trial_{i}.json")
        individuals \
            = list({hash(adapter.restore(ind.graph)): ind
                    for gen in history.generations
                    for ind in reversed(list(gen))}.values())
        evaluator(individuals)
        for ind in individuals:
            all_individuals['generated_coformers'].append(adapter.restore(ind.graph).get_smiles())
            all_individuals['orthogonal_planes'].append(abs(ind.fitness.getValues()[0]))
            all_individuals['unobstucted'].append(abs(ind.fitness.getValues()[1]))
            all_individuals['h_bond_bridging'].append(ind.fitness.getValues()[2])
            all_individuals['sa'].append((ind.fitness.getValues()[3]))

    all_individuals['drug'] = ['CN1C2=C(C(=O)N(C1=O)C)NC=N2'] * len(all_individuals['generated_coformers'])
    all_df = pd.DataFrame(data=all_individuals)

    all_df.to_csv('all_GOLEM_generated_from_cvae.csv', index=False)

    valid_df = all_df[(all_df.orthogonal_planes >= 0.332) & (all_df.unobstucted >= 0.5) & (all_df.h_bond_bridging <= 0.5) & (all_df.sa <= 3)]
    validity = len(valid_df) / len(all_df)

    duplicates = all_df.duplicated(subset='generated_coformers', keep='first').sum() / len(all_df)

    mean_sa_all = all_df.sa.mean()
    mean_sa_valid = valid_df.sa.mean()
    sa_le_3 = len(all_df[all_df.sa <= 3]) / len(all_df)

    print(f'For {exp_name_path}:\n'
          f'All generated: {len(all_df)}'
          f'Valid_generated: {len(valid_df)}'
          f'Validity: {validity}\n'
          f'Duplicates: {duplicates}\n'
          f'SA <= 3 coef: {sa_le_3}\n'
          f'Mean SA all: {mean_sa_all}\n'
          f'Mean SA valid: {mean_sa_valid}')



if __name__=='__main__':
    # #rnn
    # check_novelty('D:\Projects\Cocrystal37pip\AAAI_code\GAN\CCDC_fine-tuning\data\database_CCDC_0.csv',
    #               'D:\Projects\Cocrystal37pip\AAAI_code\GAN\ChEMBL_training//rnn_data.csv',
    #               '0',
    #               '0')

    # check_novelty('D:\Projects\Cocrystal37pip\AAAI_code\GAN\CCDC_fine-tuning\data\database_CCDC_0.csv',
    #               'D:\Projects\Cocrystal37pip\molGCT//fine_tune\moses_bench2_lat=128_epo=1111111111111_k=4_20231218.csv',
    #               '0',
    #               'mol')

    # check_novelty('/CCDC_fine-tuning/data/database_CCDC_0.csv',
    #               'D:\Projects\Cocrystal37pip\molGCT//fine_tune//10k_moses_bench2_lat=128_epo=1111111111111_k=4_20231219.csv',
    #               '0',
    #               'mol')

    check_validity(r"D:\Лаба\molecule_seacrh\cocrysals_data\results\cvae_evo")

    # check_novelty('D:\Projects\Cocrystal37pip\AAAI_code\GAN\ChEMBL_training\data\database_ChEMBL.csv',
    #               'D:\Projects\Cocrystal37pip\molGCT\moses_bench2_lat=128_epo=1111111111111_k=4_20231218.csv',
    #               '0',
    #               'mol')
