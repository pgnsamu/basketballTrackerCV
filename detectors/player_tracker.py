import numpy as np
import pandas as pd
from utils import Player, Ball

class PlayerTracker:
    def __init__(self):
        pass

    def interpolate_ball_positions(self, ball_positions: list[Ball]):
        """
        Interpola le posizioni mancanti della palla.
        """
        # Estraiamo i dati in un formato gestibile (lista di dizionari)
        ball_positions_list = [
            x.xyxy if x is not None else [np.nan, np.nan, np.nan, np.nan] 
            for x in ball_positions
        ]
        
        # Creiamo un DataFrame pandas
        df_ball_positions = pd.DataFrame(ball_positions_list, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolazione Lineare (riempie i buchi)
        df_ball_positions = df_ball_positions.interpolate()
        
        # Riempie i bordi rimasti vuoti (es. primi frame) con la tecnica 'bfill' (backfill)
        df_ball_positions = df_ball_positions.bfill()

        # Ricostruiamo la lista di oggetti Ball
        interpolated_ball_positions = []
        for i, row in df_ball_positions.iterrows():
            if np.isnan(row['x1']):
                interpolated_ball_positions.append(None)
            else:
                xyxy = np.array([row['x1'], row['y1'], row['x2'], row['y2']])
                # Creiamo un nuovo oggetto Ball interpolato (confidenza fittizia 1.0)
                interpolated_ball_positions.append(Ball(xyxy=xyxy, conf=1.0, class_id=0))

        return interpolated_ball_positions

    def interpolate_player_positions(self, player_positions: list[list[Player]]):
        """
        Interpola le posizioni dei giocatori per evitare sfarfallii (flickering).
        """
        # 1. Appiattiamo la lista per estrarre tutti i rilevamenti
        detection_list = []
        for frame_num, players_in_frame in enumerate(player_positions):
            for player in players_in_frame:
                if player.track_id is not None:
                    detection_list.append({
                        'frame': frame_num,
                        'track_id': player.track_id,
                        'x1': player.xyxy[0],
                        'y1': player.xyxy[1],
                        'x2': player.xyxy[2],
                        'y2': player.xyxy[3],
                        'class_id': player.class_id
                    })
        
        if not detection_list:
            return player_positions

        df = pd.DataFrame(detection_list)
        
        # 2. Interpoliamo separatamente per ogni Track ID
        unique_track_ids = df['track_id'].unique()
        interpolated_data = []

        for track_id in unique_track_ids:
            track_df = df[df['track_id'] == track_id]
            
            # Creiamo un indice completo dal primo all'ultimo frame in cui appare questo giocatore
            min_frame = track_df['frame'].min()
            max_frame = track_df['frame'].max()
            full_range = pd.RangeIndex(start=min_frame, stop=max_frame + 1, name='frame')
            
            # Reindicizziamo per far apparire i buchi (NaN)
            track_df = track_df.set_index('frame').reindex(full_range)
            
            # Interpoliamo i valori mancanti
            track_df[['x1', 'y1', 'x2', 'y2']] = track_df[['x1', 'y1', 'x2', 'y2']].interpolate()
            
            # Riportiamo class_id e track_id (che diventano NaN nel reindex)
            track_df['class_id'] = track_df['class_id'].ffill().bfill()
            track_df['track_id'] = track_id # Ripristiniamo l'ID
            
            # Salviamo
            track_df = track_df.reset_index()
            interpolated_data.append(track_df)

        # Uniamo tutto
        df_interpolated = pd.concat(interpolated_data)

        # 3. Ricostruiamo la struttura originale list[list[Player]]
        # Inizializziamo lista vuota della lunghezza giusta
        output_player_positions = [[] for _ in range(len(player_positions))]

        for i, row in df_interpolated.iterrows():
            if np.isnan(row['x1']): continue # Sicurezza
            
            frame_idx = int(row['frame'])
            
            # Ricreiamo il giocatore
            player = Player(
                track_id=int(row['track_id']),
                xyxy=np.array([row['x1'], row['y1'], row['x2'], row['y2']]),
                conf=0.99, # Confidenza fittizia per i punti interpolati
                class_id=int(row['class_id'])
            )
            
            if frame_idx < len(output_player_positions):
                output_player_positions[frame_idx].append(player)

        return output_player_positions