G:\My Drive\cs231n\project\cs231n claude sessions a

1, remove OCR landmarks in process
2. do a bdget calculation using Gemini Pro
3. indicate the video downloading code and running command
4. indicate the the code running command for frame extraction, I want to run it 
5.in 3.3, show how landmarks_zurich_osm.json is extracted,show the running command, how bbox determined, what is the purpose of  zurich_landmark_gps.py, what is the purpose of scenery_pois.py 
Gemma POI scan, show the input output and code  how was the fixed 26 candidate POI list refer to, what is the relationship of the 3 tables 
no need to rerun the POI scan using gemini
6. in 3.4 the bbox, should it cater for a little be larger, in case the POI is just on the bbox, there might be route  outside ofth bbox in the video taken right?
7. in 3.5 DINOv2 match against the Street View index - what is the value for min-sim, we can run a a few rounds, and then pick the one with most pictures and best quality, with a sanity check on 10% of the data
then remove OCR LANDMARKS,what is the breaking criteria used for DINOv2 and VLM geo-localization? - this we might need to use gemini-fast to run, need to see the code for input and output. how was the heading angle concluded?
can use the sample test to run first, just to check 
8.in 3.6 I have a question regarding this, can we use the GPS photos as part of the GT to do instruction tuning annotation?
9. 3.7 this time we use gemini 2.5 pro for the annoation, let me know the code and we can run 5 smaples first. 
10. 3.8 yes we need to keep the schema, I think ideally after you have mapped the images to GPS, i would want to see route map for the 8 ppl, since you are able to get the GPS, it should be possible to get that visualized
11. to answer 4, the above points have already mentioned
12. for 6 experiments & training what was the experiment settings can you describe? 
13. for visulalization, there are a few items:
- the list of POIs extracted by gemma from the video frame on map, if possible add a picture of POI on the map 
- the highlight of street view photos bought online 
- after mapping the highlght of the mapped video frames 
- the route derived from the images 
- the visulization of Q&A, based on the answer, we can do some sanity  check of the instruction tuning.
now you can start developing and note to keep a neat code structure and use relative path, while developing update the dev.md to answer my questions above one by one 