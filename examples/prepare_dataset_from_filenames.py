from pathlib import Path

import numpy as np
import pandas as pd
from obspy import read
from obspy import UTCDateTime as utc
from pandas import DataFrame

from yews.datasets.utils import get_files_under_dir
from yews.datasets.utils import read_frame_obspy

def retrieve_info_from_path(path):
    name = path.name
    event_id = path.parent.name
    event_time = float(utc(event_id))
    lat = None
    lon = None
    dep = None
    phase_info, event_info = name.split('_')
    m = event_info.replace('.SAC','')
    phase_info = phase_info.split('.')
    phase, station, channel = phase_info[:3]
    arrival_list = phase_info[3:]
    arrival_list.insert(6, '.')
    arrival_list.insert(3, 'T')
    arrival=float(utc(''.join(arrival_list)))

    return {
        'id': event_id,
        'origin': event_time,
        'latitude': lat,
        'longitude': lon,
        'depth': dep,
        'magnitude': m,
        'phase': phase,
        'arrival': arrival,
        'station': station,
        'channel': channel,
        'path': path,
    }

def retrieve_station_info(path):
    station_name = path.name.split('.')[1]
    header = read(str(path), headonly=True)[0].stats['sac']
    return {
        'station': station_name,
        'latitude': float(header['stla']),
        'longitude': float(header['stlo']),
    }

def path2pattern(path):
    root = path.parent
    name = path.name
    name = name.split('.')
    name[2] = '*'
    name = '.'.join(name)
    return root / name


if __name__ == '__main__':
    ###########################################################################
    #
    #                   Construct info tables from filenames
    #
    ###########################################################################

    paths = [Path(a) for a in ['events.csv', 'phases.csv', 'stations.csv']]
    if all([a.exists() for a in paths]):
        events = pd.read_csv(paths[0], index_col='id')
        phases = pd.read_csv(paths[1])
        events = pd.read_csv(paths[2], index_col='station')
    else:
        files = get_files_under_dir('/data/ok_new','**/*.SAC')
        info = map(retrieve_info_from_path, files)
        trace_table = DataFrame(info)

        # retrive event table
        events = trace_table[['id','origin','latitude','longitude', 'depth',
                                   'magnitude']].drop_duplicates('id')
        events.set_index('id', inplace=True, verify_integrity=True)
        events.to_csv('events.csv')

        # check if there is inconsistent event information
        #duplicated_events = events.duplicated('id')
        #if duplicated_events.any():
        #    raise ValueError(f'Inconsistent events are as following:\n'
        #                     f'{events[duplicated_events]}')


        # retrive phase table
        phases = trace_table[['id','phase','arrival','station', 'path']].drop_duplicates(subset=['id', 'phase', 'arrival', 'station'])
        phases['path'] = phases[['path']].applymap(path2pattern)
        phases.reset_index(drop=True, inplace=True)
        phases.to_csv('phases.csv')

        # retrive station table
        stations = trace_table[['station', 'path']].drop_duplicates('station')
        station_info = map(retrieve_station_info, stations['path'])
        stations = DataFrame(station_info)
        stations.set_index('station', inplace=True, verify_integrity=True)
        stations.to_csv('stations.csv')


    ###########################################################################
    #
    #                 Construct NPY for samples and targets
    #
    ###########################################################################

    samples_list = []
    targets_list = []
    for index, row in phases.iterrows():
        if index % 1000 == 0:
            print(index)
        try:
            data = read_frame_obspy(str(row['path']))
        except ValueError:
            print(f"Waveform #{index} is broken. Skipped.")
            continue
        if data.shape != (3, 9600):
            print(f"Phase #{index} is invalid.")
            continue # skip broken data
        phase = row['phase']
        samples_list.append(data[:, 4600:5400])
        targets_list.append([phase, index])
        if phase == 'P':
            samples_list.append(data[:, 1600:2400])
            targets_list.append(['N', index])
        elif phase == 'S':
            samples_list.append(data[:, 7800:8600])
            targets_list.append(['N', index])
        else:
            print(f"{index} has a invalid phase {phase}.")
            continue # skip unknown phases
    samples = np.stack(samples_list)
    targets = np.stack(targets_list)

    print(samples.shape)
    print(targets.shape)

    np.save('samples.npy', samples)
    np.save('targets.npy', targets)
