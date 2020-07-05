#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2020 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
from CESP.neural_network.architecture.unet.standard import Architecture as UNet_standard
from CESP.neural_network.architecture.unet.plain import Architecture as UNet_plain
from CESP.neural_network.architecture.unet.residual import Architecture as UNet_residual
from CESP.neural_network.architecture.unet.dense import Architecture as UNet_dense
from CESP.neural_network.architecture.unet.multiRes import Architecture as UNet_multiRes
from CESP.neural_network.architecture.unet.compact import Architecture as UNet_compact
