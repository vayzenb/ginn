#!/bin/bash

module load fsl-6.0.3



#subj_list="pixar155"
exp="/lab_data/behrmannlab/vlad/ginn/fmri/aeronaut"
roi="V3ab V4 PPC APC"
roi="left_hemisphere right_hemisphere"
parcelType=mruczek_parcels
#parcelType=infant_parcels
mniBrain=$FSLDIR/data/standard/MNI152_T1_1mm.nii.gz
#mniBrain=/user_data/vayzenbe/GitHub_Repos/fmri/roiParcels/$parcelType/infantTemplate.nii.gz
anat=$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz
anat=/lab_data/behrmannlab/vlad/ginn/fmri/aeronaut/derivatives/rois/mni_mask.nii.gz

parcelDir=/user_data/vayzenbe/GitHub_Repos/fmri/roiParcels/$parcelType
#parcelDir=/lab_data/behrmannlab/vlad/ginn/fmri/aeronaut/derivatives/rois
studyDir=${exp}/derivatives/rois
roiDir=${exp}/derivatives/rois

mkdir $roiDir
echo $anat
#Create registration matrix from MNI to anatomy
flirt -in $mniBrain -ref $anat -omat $roiDir/stand2exp.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12	

for rr in $roi
do

	#Register binary files to anatomical
	#flirt -in $parcelDir/l${rr}.nii.gz -ref $anat -out $roiDir/l${rr}.nii.gz -applyxfm -init $roiDir/stand2exp.mat -interp trilinear
	#flirt -in $parcelDir/r${rr}.nii.gz -ref $anat -out $roiDir/r${rr}.nii.gz -applyxfm -init $roiDir/stand2exp.mat -interp trilinear
	flirt -in $parcelDir/${rr}.nii.gz -ref $anat -out $roiDir/${rr}.nii.gz -applyxfm -init $roiDir/stand2exp.mat -interp trilinear
done


