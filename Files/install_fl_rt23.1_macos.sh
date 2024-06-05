#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
#
#
# Script to create a conda environment and install ibm-watson-machine-learning with 
# the dependencies required for Federated Learning on MacOS using Runtime 22.2 or 23.1.
# The first argument is the runtime.
# The second argument is the name of the conda environment to be created.
# 
# Note: This script requires miniforge to be installed for conda.
# 

usage="Usage: ./install_fl_macos.sh <fl-rt22.2-py3.10 | fl-rt23.1-py3.10> <conda_env_name>"

export RUNTIME=$1
ENAME=$2

arch=$(uname -m)
os=$(uname -s)

if [[ $# -ne 2 ]]
then
   echo $usage
   exit
fi

case $RUNTIME in

  fl-rt22.2-py3.10)
    PYTHON_VERSION="3.10"
    export TF_VERSION="2.9.2"
    ;;

  fl-rt23.1-py3.10)
    PYTHON_VERSION="3.10"
    export TF_VERSION="2.12.0"
    ;;

  *)
    echo $usage
    exit

esac

for needed in conda pip python
do
  if ! command -v $needed &> /dev/null
  then
     echo "$needed could not be found"
     exit
  fi
done


source $CONDA_PREFIX/etc/profile.d/conda.sh
conda create -y -n ${ENAME} python=${PYTHON_VERSION}
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
import os

package = 'ibm-watson-machine-learning'
extra   = os.environ['RUNTIME']
tf_ver  = os.environ['TF_VERSION']
extra_  = extra.replace('.','-')
extra_s = '; extra == "{}"'    
remove  = None
add     = []

if platform.system() == "Darwin" and platform.processor() == "arm":
    remove  = 'tensorflow'
    add     = ['tensorflow-macos==' + tf_ver]

pkgs = pkg_resources.working_set.by_key[package].requires(extras=[extra])
pkgs = [ p.__str__().removesuffix(extra_s.format(extra)).removesuffix(extra_s.format(extra_)) for p in pkgs if ( extra in p.__str__() or extra_ in p.__str__() ) and ( not remove or remove not in p.__str__() )]

print( "Installing standard packages for {}[{}]:{}".format(package,extra,pkgs) )
print( "Installing additional packages:{}".format(add) )

cmd = [ 'pip', 'install'] + add + pkgs

subprocess.run( cmd )
EOF
