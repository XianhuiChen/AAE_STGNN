import os
import pandas as pd
import est

def read_raw_data():
    df = pd.read_csv('../data/tmp.csv')
    df_label = pd.read_csv(r'../data/ZCTA dataset - First visit UDT.csv')
    return df, df_label

def analyze_raw_data():
    first_visit = group.drop_duplicates(subset=['PATIENT_HASH_KEY'], keep='first')
    fentanyl_mask = (first_visit['FENTANYL'] == 1) & (first_visit['RX_FENTANYL'] == -1)
    fent_group = first_visit[fentanyl_mask]

    ## fent.meth
    meth_group = fent_group[fent_group['METHAMPHETAMINE'] != 0]
    # result['fent.meth'] = len(meth_group)
    # meth_pos = meth_group[
    #     (meth_group['METHAMPHETAMINE'] == 1) & (meth_group['RX_METHAMPHETAMINE'] == -1)
    #     ]
    # result['fent.meth.pos'] = len(meth_pos)
    # result['fent.meth.per'] = result['fent.meth.pos'] / result['fent.meth'] if result['fent.meth'] > 0 else 0


def get_fentanyl_combo_stats(group):
    result = {}
    first_visit = group.drop_duplicates(subset=['PATIENT_HASH_KEY'], keep='first')
    fentanyl_mask = (first_visit['FENTANYL'] == 1) & (first_visit['RX_FENTANYL'] == -1)
    fent_group = first_visit[fentanyl_mask]

    ## fent.meth
    meth_group = fent_group[fent_group['METHAMPHETAMINE'] != 0]
    result['fent.meth'] = len(meth_group)
    meth_pos = meth_group[
        (meth_group['METHAMPHETAMINE'] == 1) & (meth_group['RX_METHAMPHETAMINE'] == -1)
        ]
    result['fent.meth.pos'] = len(meth_pos)
    result['fent.meth.per'] = result['fent.meth.pos'] / result['fent.meth'] if result['fent.meth'] > 0 else 0

    ## fent.cocaine
    coc_group = fent_group[fent_group['COCAINE'] != 0]
    result['fent.cocaine'] = len(coc_group)
    coc_pos = coc_group[
        (coc_group['COCAINE'] == 1) & (coc_group['RX_COCAINE'] == -1)
        ]
    result['fent.cocaine.pos'] = len(coc_pos)
    result['fent.cocaine.per'] = result['fent.cocaine.pos'] / result['fent.cocaine'] if result[
                                                                                            'fent.cocaine'] > 0 else 0

    ## fent.stim = fent.meth + fent.cocaine
    result['fent.stim'] = result['fent.meth'] + result['fent.cocaine']
    result['fent.stim.pos'] = result['fent.meth.pos'] + result['fent.cocaine.pos']
    result['fent.stim.per'] = result['fent.stim.pos'] / result['fent.stim'] if result['fent.stim'] > 0 else 0

    ## fent.heroin
    heroin_group = fent_group[fent_group['Heroin'] != 0]
    result['fent.heroin'] = len(heroin_group)
    heroin_pos = heroin_group[heroin_group['Heroin'] == 1]
    result['fent.heroin.pos'] = len(heroin_pos)
    result['fent.heroin.per'] = result['fent.heroin.pos'] / result['fent.heroin'] if result[
                                                                                         'fent.heroin'] > 0 else 0

    return pd.Series(result)




def main():
    df, df_label = read_raw_data()
    fent_stats = df.groupby(['test_year', 'test_month', 'zcta']).apply(get_fentanyl_combo_stats).reset_index()
    fent_stats = fent_stats.rename(columns ={
        'test_year': 'Year',
        'test_month': 'Month',
        'zcta': 'ZCTA'
        })
    # df = df.concat([df_label, feat_stats])
    print(fent_stats)
    return

    dfl = df_label[[
              'tm',
              'Year',
              'Month',
              'ZCTA',
              'opioid.overdose'
            ]]
    print(dfl[dfl['ZCTA'] == 43001])
    print(df_label['opioid.overdose'].unique())

    nb_count_dict = {}
    for n in df_label['opioid.overdose']:
        nb_count_dict[n] = nb_count_dict.get(n, 0) + 1
    for i in sorted(nb_count_dict):
        print(i, nb_count_dict[i])

    # fent_stats = df.groupby(['collect_year', 'collect_month', 'zcta']).apply(get_fentanyl_combo_stats).reset_index()
    # print(df.columns)
    # print(df.shape)
    # print(fent_stats.columns)
    # print(fent_stats.shape)

if __name__ == '__main__':
    main()
    # est.estimate_smape()