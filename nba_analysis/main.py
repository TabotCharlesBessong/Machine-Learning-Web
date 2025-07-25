
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer, TeamBallControlDrawer, PassInterceptionDrawer, CourtKeypointDrawer, TacticalViewDrawer
from team_assigner import TeamAssigner
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from court_keypoint_detector import CourtKeypointDetector
from tactical_view_converter import TacticalViewConverter

def main():
  # read a video file
  video_frames = read_video("input_videos/video_3.mp4")
  
  # Initialize the player tracker with the YOLO model path
  player_tracker = PlayerTracker("models/players/best.pt")
  ball_tracker = BallTracker("models/balls/best.pt")
  
  # Initialize the court keypoint detector
  court_keypoint_detector = CourtKeypointDetector("models/court/best.pt")
  
  # run trackers
  player_tracks = player_tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path="stubs/player_track_stubs3.pkl")
  ball_tracks = ball_tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path="stubs/ball_track_stubs3.pkl")
  
  # remove bad ball detections
  player_tracks = ball_tracker.remove_wrong_detections(player_tracks)
  # interpolate ball positions
  ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)
  
  # Get court keypoints
  court_keypoints = court_keypoint_detector.get_court_keypoints(video_frames,read_from_stub=True, stub_path="stubs/court_keypoint_stubs3.pkl")
  
  # print("Court Keypoints Detected:", court_keypoints)
  
  # assign teams to players
  team_assigner = TeamAssigner()
  player_teams = team_assigner.get_player_teams_across_frames(video_frames,player_tracks,read_from_stub=True,stub_path="stubs/player_team_stubs3.pkl")
  
  # ball acquisition detection
  ball_aquisition_detector = BallAquisitionDetector()
  ball_aquisition = ball_aquisition_detector.detect_ball_possession(player_tracks, ball_tracks)
  
  # detect passes and interceptions
  pass_and_interception_detector = PassAndInterceptionDetector()
  passes = pass_and_interception_detector.detect_passes(ball_aquisition, player_teams)
  interceptions = pass_and_interception_detector.detect_interceptions(ball_aquisition, player_teams)
  
  # Tactical view conversion
  tactical_view_converter = TacticalViewConverter(court_image_path="./images/basketball_court.png")
  court_keypoints = tactical_view_converter.validate_keypoints(court_keypoints)
  
  # print("Passes Detected:", passes)
  # print("Interceptions Detected:", interceptions)
  
  # print(ball_aquisition)
  
  # print("Player Teams Assigned:", player_teams)


  # draw output
  # Initialize the player tracks drawer
  player_tracks_drawer = PlayerTracksDrawer() 
  ball_tracker_drawer = BallTracksDrawer()
  team_ball_control_drawer = TeamBallControlDrawer()
  pass_and_interception_drawer = PassInterceptionDrawer()
  court_keypoint_drawer = CourtKeypointDrawer()
  tactical_view_drawer = TacticalViewDrawer()
  
  # drawe object tracks on video frames
  output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks,player_teams,ball_aquisition)
  output_video_frames = ball_tracker_drawer.draw(output_video_frames, ball_tracks)
  
  # draw team ball control statistics
  output_video_frames = team_ball_control_drawer.draw(output_video_frames, player_teams, ball_aquisition)
  output_video_frames = pass_and_interception_drawer.draw(output_video_frames, passes, interceptions)
  
  # draw court keypoints
  output_video_frames = court_keypoint_drawer.draw(output_video_frames, court_keypoints)
  
  # tactical view conversion
  output_video_frames = tactical_view_drawer.draw(
    output_video_frames,
    tactical_view_converter.court_image_path,
    width=tactical_view_converter.width,
    height=tactical_view_converter.height,
    tactical_court_keypoints=court_keypoints,
  )
  
  # save video frame
  save_video(output_video_frames,"output_videos/track/output_video_3.avi")
if __name__ == "__main__":
  main()