#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
#
#
# Script to create a conda environment and install ibm-watson-machine-learning with 
# the dependencies required for Federated Learning on MacOS using Runtime 22.2.
# The name of the conda environment to be created is passed as the first argument.
# 
# Note: This script requires miniforge to be installed for conda.
# 

usage=". install_fl_rt22.2_macos.sh conda_env_name"

arch=$(uname -m)
os=$(uname -s)

if (($# < 1)) 
then
   echo $usage
   exit
fi

ENAME=$1

source $CONDA_PREFIX/etc/profile.d/conda.sh
conda create -y -n ${ENAME} python=3.10
conda activate ${ENAME}
if [[ $? -ne 0 ]]; then
    echo "Error Activating Conda Environment: ${ENAME}"
    exit 1
fi

pip install ibm-watson-machine-learning

if [ "$os" == "Darwin" -a "$arch" == "arm64" ]
then
   conda install -y -c apple tensorflow-deps
fi 

python - <<EOF
import pkg_resources
import platform
import subprocess

package = 'ibm-watson-machine-learning'
extra   = 'fl-rt22.2-py3.10'     
extra_  = extra.replace('.','-')
extra_s = '; extra == "{}"'    
remove  = None
add     = []

if platform.system() == "Darwin" and platform.processor() == "arm":
    remove  = 'tensorflow'
    add     = ['tensorflow-macos==2.9.2']

pkgs = pkg_resources.working_set.by_key[package].requires(extras=[extra])
pkgs = [ p.__str__().removesuffix(extra_s.format(extra)).removesuffix(extra_s.format(extra_)) for p in pkgs if ( extra in p.__str__() or extra_ in p.__str__() ) and ( not remove or remove not in p.__str__() )]

print( "Installing standard packages for {}[{}]:{}".format(package,extra,pkgs) )
print( "Installing additional packages:{}".format(add) )

cmd = [ 'pip', 'install'] + add + pkgs

subprocess.run( cmd )
EOF
