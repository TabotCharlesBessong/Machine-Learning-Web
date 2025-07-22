
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer

def main():
  # read a video file
  video_frames = read_video("input_videos/video_1.mp4")
  
  # Initialize the player tracker with the YOLO model path
  player_tracker = PlayerTracker("models/players/best.pt")
  ball_tracker = BallTracker("models/balls/best.pt")
  
  # run trackers
  player_tracks = player_tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path="stubs/player_track_stubs.pkl")
  ball_tracks = ball_tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path="stubs/ball_track_stubs.pkl")
  
  # remove bad ball detections
  player_tracks = ball_tracker.remove_wrong_detections(player_tracks)


  # draw output
  # Initialize the player tracks drawer
  player_tracks_drawer = PlayerTracksDrawer() 
  ball_tracker_drawer = BallTracksDrawer()
  # drawe object tracks on video frames
  output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks)
  output_video_frames = ball_tracker_drawer.draw(output_video_frames, ball_tracks)
  # save video frame
  save_video(output_video_frames,"output_videos/track/output_video_1.avi")
if __name__ == "__main__":
  main()