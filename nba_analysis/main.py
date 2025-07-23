
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer
from team_assigner import TeamAssigner
from ball_aquisition import BallAquisitionDetector

def main():
  # read a video file
  video_frames = read_video("input_videos/video_2.mp4")
  
  # Initialize the player tracker with the YOLO model path
  player_tracker = PlayerTracker("models/players/best.pt")
  ball_tracker = BallTracker("models/balls/best.pt")
  
  # run trackers
  player_tracks = player_tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path="stubs/player_track_stubs.pkl")
  ball_tracks = ball_tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path="stubs/ball_track_stubs.pkl")
  
  # remove bad ball detections
  player_tracks = ball_tracker.remove_wrong_detections(player_tracks)
  # interpolate ball positions
  ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)
  
  # assign teams to players
  team_assigner = TeamAssigner()
  player_teams = team_assigner.get_player_teams_across_frames(video_frames,player_tracks,read_from_stub=True,stub_path="stubs/player_team_stubs.pkl")
  
  # ball acquisition detection
  ball_aquisition_detector = BallAquisitionDetector()
  ball_aquisition = ball_aquisition_detector.detect_ball_possession(player_tracks, ball_tracks)
  
  # print(ball_aquisition)
  
  # print("Player Teams Assigned:", player_teams)


  # draw output
  # Initialize the player tracks drawer
  player_tracks_drawer = PlayerTracksDrawer() 
  ball_tracker_drawer = BallTracksDrawer()
  # drawe object tracks on video frames
  output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks,player_teams,ball_aquisition)
  output_video_frames = ball_tracker_drawer.draw(output_video_frames, ball_tracks)
  # save video frame
  save_video(output_video_frames,"output_videos/track/output_video_2.avi")
if __name__ == "__main__":
  main()