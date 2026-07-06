# BSc-Thesis

Overview of the files:
1. regal_dnn_training_code.py: this is the model that we built to compute distances to galaxies. Requires input .npz cache
2. add_point_sources_test.py: this is a robustness test where we inject point sources
3. artpop_trainingset_02pix_1e8.py: this is an example artpop simulation code for 10^8 stars per image
4. augment_rotate.py: this adds noise to images generated from artpop (that are already psf-convolved), and augments the dataset with a 90 degree rotation
5. compare_good_vs_bad.py: compares the good and bad galaxies per stripe (angular size degeneracy issue)
6. crop_residuals.py: crops all residuals to 0.84 times effective radius
7. make_residuals.py: make residuals with (data-model)/sqrt(model)
8. unseen_dist_code.py: robustness testing with unseen distances
9. eval_hst.py: evaluates success of robustness test with real galaxies (HST data)
10. eval_contamination.py: evaluates success of robustness test with added point sources (number 2 in this list)
11. eval_unseen.py: evaluates success of robustness test with unseen distances (number 8 in this list)
12. fits_to_numpy.py: converts directory with .fits files into a .npz cache
13. hst_preprocess.py: process HST images to make them into residuals to use as input
