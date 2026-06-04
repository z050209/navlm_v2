file:///C:/Users/z0502/Desktop/cs231n/navlm_v2/viz/trusted_frames_grid_50.html
file:///C:/Users/z0502/Desktop/cs231n/navlm_v2/viz/heading_check.html

pre: use VLM to scan thor.05 and smaller than 0.2 tugh the frames and get the bbox
buy the GPS photos and calculate DINOV2
get the DINO V2 results pass through VLM again 

 - All apps: https://modal.com/apps/z050209
  - navlm-eval app runs: https://modal.com/apps/z050209/main/navlm-eval
  - navlm-train app runs: https://modal.com/apps/z050209/main/navlm-train
  - Volumes: https://modal.com/storage/z050209/main/navlm-data
  - Logs of a specific function call appear under each app's "Functions" tab, including live stdout/stderr

for each the photos pass dinov2 
  1. step1 curate a list of attrachtions, POI, landmarks can be derived from GPS and OSM label, file name GPS_GEO.jsonl
  2. step2 curate a list of attrachtions, POI, landmarks can be derived from GPS and OSM label, file name VLM_GEO.jsonl
  3. for cosine similarity > 0.75 
  step3 compare list in step  and step 2, note the naming format can be different, give more space for the affiliation, there needs to be at least one coincide between the two lists, then we say they match, we don't do neibhbourhood check, file_nam GPS_VLM_GEO.jsonl 
  4. step4 visualize the mapping of image with GPS as file:///C:/Users/z0502/Desktop/cs231n/navlm_v2/viz/gps_recovery_full_grid_vlm_agreed_50.html, random sample 30 of them I want see the result a2_vlmagreed.html

 if the gap between the top1 direction and the next direction for cosine simiary is larger than 0.2, we will use    
  the top1 results, if smalled than top1 together with top2 as you calcualted above

  can you draw  these 89 slots on map to a2_mapped_GPS_spot.html

  4. step4 check again whether there is still miss mathc issu e


  after solving the mapping issue 
  get the list of frames that are specific for the shortlisted attractions we named 
  make them as the target attractions -- keep track on how many frames are representing  target locations
  let VLM to annotate the any frame as current location -- keep track on how man are current locations

  this use OSM to create the route first 
  ❯ Pure random, Re-roll the band if duplicate, 80/10/10, 3 per frame, use Recommended hybrid method, please do up the route gt now
 file:///C:/Users/z0502/Desktop/cs231n/navlm_v2/viz/a2_route_gt.html 
then strategize the each image ask 3 questions to the attraction within 80% with 500m. 10 % 500 -1000m 10 5 1000M -1500M

next, with the heading and osm route as input revise the system prompt for the VLM gemini pro to generate instruction tuning dataset, think about how to evaluate the accuracy, we have the criteria of format and direction accuracy. 
we still only consider the first verb replied for direction 
redesing the experiment of zero thought, what is explict and implicit, when to hide the information of heading with CoT 
 
next, send for annotation 

next, visualize some of the annotation samples and do the QC into a2_viz_sft.html

next, strategize the dataset speration for train/test/val 

next, send for zero thought and finueting, I think the zero thought should also have three mode as sft as mentioned above 
