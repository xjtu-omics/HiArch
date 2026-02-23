#!/bin/bash



#sps matrix should end with .mtx
#.mtx and window.bed file are produced by hic2sps.py

#start your work
#Set your work folder like workdir/species/, in workdir you can make one more species dir
#Then create sps_mtx dir in your "species" , where all samples' samples_normalized.mtx and samples.window.bed files produced by hic2sps.py locate

paraFile=$1
source $paraFile
cd
for species in "$basepath"/*
do

    echo $(basename ${species})
    for sample in $species/sps_mtx/*_normalized.mtx
    do
        name=$(basename $sample .mtx)
        echo "processing $name"

        step="normDis"
        echo "${step} will begin"
        if [ ! -d ${species}/result/${step}/mtx ];then
            echo "create ${species}/result/${step}/mtx"
            mkdir -p ${species}/result/${step}/mtx
        else
            echo "${species}/result/${step}/mtx already exists"
        fi

        if [ ! -d ${species}/result/${step}/figure ];then
            echo "create ${species}/result/${step}/figure"
            mkdir -p ${species}/result/${step}/figure
        else
            echo "${species}/result/${step}/figure already exists"
        fi
    
        ${python_path} ${script_dir}/NormDis.py \
        -f ${species}/sps_mtx/${name}.mtx \
        -w ${species}/sps_mtx/${name/_normalized/}.window.bed \
        -o ${species}/result/${step}/mtx/${name} \
        -fo ${species}/result/${step}/figure/${name} \
        -df ${normdis_df} \
        -cmd ${normdis_cmd}

        if [ $? -ne 0 ]; then
        echo "Error occurred while running $step for $name. Exiting..."
        exit 1  
        fi


        pre_step="normDis"
        step="correctMap"
        echo "${pre_step} finished, ${step} will begin"
        if [ ! -d ${species}/result/${step}/mtx ];then
            echo "create ${species}/result/${step}/mtx"
            mkdir -p ${species}/result/${step}/mtx
        else
            echo "${species}/result/${step}/mtx already exists"
        fi

        if [ ! -d ${species}/result/${step}/figure ];then
            echo "create ${species}/result/${step}/figure"
            mkdir -p ${species}/result/${step}/figure
        else
            echo "${species}/result/${step}/figure already exists"
        fi

        ${python_path} ${script_dir}/CorrectMap.py \
        -f ${species}/result/${pre_step}/mtx/${name}.de_ode.mtx \
        -w ${species}/result/${pre_step}/mtx/${name}.window.bed \
        -o ${species}/result/${step}/mtx/${name} \
        -fo ${species}/result/${step}/figure/${name} \
        -ac ${correctmap_ac} \
        -drc ${correctmap_drc}

        if [ $? -ne 0 ]; then
        echo "Error occurred while running $step for $name. Exiting..."
        exit 1  
        fi

        pre_step="correctMap"
        step="checkerBoard"
        echo "${pre_step} finished, ${step} will begin"
        if [ ! -d ${species}/result/${step}/mtx ];then
            echo "create ${species}/result/${step}/mtx"
            mkdir -p ${species}/result/${step}/mtx
        else
            echo "${species}/result/${step}/mtx already exists"
        fi

        if [ ! -d ${species}/result/${step}/figure ];then
            echo "create ${species}/result/${step}/figure"
            mkdir -p ${species}/result/${step}/figure
        else
            echo "${species}/result/${step}/figure already exists"
        fi
        
        
        ${python_path} ${script_dir}/Checkerboard.py \
        -f ${species}/result/${pre_step}/mtx/${name}.clean_de_ode.mtx \
        -w ${species}/result/${pre_step}/mtx/${name}.clean_window.bed \
        -o ${species}/result/${step}/mtx/${name}.checkerBoard \
        -sd ${checkboard_sd} \
        -fo ${species}/result/${step}/figure/${name}

        if [ $? -ne 0 ]; then
        echo "Error occurred while running $step for $name Exiting..."
        exit 1  
        fi


        pre_step="correctMap"
        step="globalFolding_s1"
        echo "${pre_step} finished, ${step} will begin"
        if [ ! -d ${species}/result/${step}/mtx ];then
            echo "create ${species}/result/${step}/mtx"
            mkdir -p ${species}/result/${step}/mtx
        else
            echo "${species}/result/${step}/mtx already exists"
        fi

        if [ ! -d ${species}/result/${step}/figure ];then
            echo "create ${species}/result/${step}/figure"
            mkdir -p ${species}/result/${step}/figure
        else
            echo "${species}/result/${step}/figure already exists"
        fi

        ${python_path} ${script_dir}/GF_S1_get_center.py \
        -f ${species}/result/${pre_step}/mtx/${name}.clean_de_ode.mtx \
        -w ${species}/result/${pre_step}/mtx/${name}.clean_window.bed \
        -o ${species}/result/${step}/mtx/${name} \
        -am ${GF_S1_am} \
        -ac ${GF_S1_ac} \
        -ue ${GF_S1_ue} \
        -fo ${species}/result/${step}/figure/${name}
        
        if [ $? -ne 0 ]; then
        echo "Error occurred while running $step for $name. Exiting..."
        exit 1  
        fi
        echo "${step} finished"


        pre_step="globalFolding_s1"
        step="globalFolding_s2"
        if [ ! -d ${species}/result/${step}/mtx ];then
            echo "create ${species}/result/${step}/mtx"
            mkdir -p ${species}/result/${step}/mtx
        else
            echo "${species}/result/${step}/mtx already exists"
        fi

        if [ ! -d ${species}/result/${step}/figure ];then
            echo "create ${species}/result/${step}/figure"
            mkdir -p ${species}/result/${step}/figure
        else
            echo "${species}/result/${step}/figure already exists"
        fi

        ${torch_path} ${script_dir}/GF_S2_get_score.py \
        -f ${species}/result/${pre_step}/mtx/${name}.filt_ode.mtx \
        -w ${species}/result/${pre_step}/mtx/${name}.window.bed \
        -af ${species}/result/${pre_step}/mtx/${name}.anchors.txt \
        -o ${species}/result/${step}/mtx/${name} \
        -fo ${species}/result/${step}/figure/${name}
        
        if [ $? -ne 0 ]; then
        echo "Error occurred while running $step for $name. Exiting..."
        exit 1  
        fi
        echo "${step} finished"

    done
done