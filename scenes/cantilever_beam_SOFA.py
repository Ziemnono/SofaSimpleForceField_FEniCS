# Required import for python
import Sofa
import SofaCaribou


# Choose in your script to activate or not the GUI
USE_GUI = True


def main():
    import SofaRuntime
    import Sofa.Gui
    SofaRuntime.importPlugin("SofaOpenglVisual")
    SofaRuntime.importPlugin("SofaImplicitOdeSolver")
    SofaRuntime.importPlugin("SofaLoader")

    root = Sofa.Core.Node("root")
    createScene(root)
    Sofa.Simulation.init(root)

    if not USE_GUI:
        for iteration in range(10):
            Sofa.Simulation.animate(root, root.dt.value)
    else:
        Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
        Sofa.Gui.GUIManager.createGUI(root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1080, 1080)
        Sofa.Gui.GUIManager.MainLoop(root)
        Sofa.Gui.GUIManager.closeGUI()


def createScene(root):
    root.addObject('DefaultVisualManagerLoop')
    root.addObject('DefaultAnimationLoop')

    root.addObject('VisualStyle', displayFlags="showForceFields showBehaviorModels")
    root.addObject('RequiredPlugin', pluginName="SofaOpenglVisual SofaSimpleForcefield SofaBaseMechanics SofaBaseTopology SofaSparseSolver SofaImplicitOdeSolver SofaTopologyMapping SofaBoundaryCondition SofaEngine")

    root.addObject('RegularGridTopology', name="grid", min="-7.5 -7.5 0", max="7.5 7.5 80", n="9 9 21")
    root.addObject('StaticSolver', newton_iterations="25", relative_correction_tolerance_threshold="1e-15", relative_residual_tolerance_threshold="1e-10", printLog="1")
    root.addObject('SparseLDLSolver', template="CompressedRowSparseMatrixMat3x3d")

    root.addObject('MechanicalObject', name="mo", src="@grid")
    root.addObject('TetrahedronSetTopologyContainer', name="topology")
    root.addObject('TetrahedronSetTopologyModifier')
    root.addObject('Hexa2TetraTopologicalMapping', input="@grid", output="@topology", swapping="1")

    root.addObject('SVKElasticForcefield_SOFA', youngModulus="3000", poissonRatio="0.3", topology="@topology")
    # root.addObject('SaintVenantKirchhoffMaterial', young_modulus=3000, poisson_ratio=0.3)
    # root.addObject('HyperelasticForcefield', printLog=True)

    root.addObject('BoxROI', name="fixed_roi", box="-7.5 -7.5 -0.9 7.5 7.5 0.1")
    root.addObject('FixedConstraint', indices="@fixed_roi.indices")
    root.addObject('BoxROI', name="top_roi", box="-7.5 -7.5 79.9 7.5 7.5 80.1")
    root.addObject('ConstantForceField', force="0 -10 0", indices="@top_roi.indices")

    # import meshio
    # beam_p1 = meshio.read("./beam_p1.vtu")


    # root.addObject('StaticSolver', newton_iterations="25", relative_correction_tolerance_threshold="1e-15", relative_residual_tolerance_threshold="1e-10", printLog="1")
    # root.addObject('SparseLDLSolver', template="CompressedRowSparseMatrixMat3x3d")

    # root.addObject('MechanicalObject', name="mo", position=beam_p1.points.tolist())
    # root.addObject('CaribouTopology', name='topology', template='Tetrahedron', indices=beam_p1.cells_dict['tetra'].tolist())
    # # root.addObject('TetrahedronSetTopologyContainer', name="topology")
    # # root.addObject('TetrahedronSetTopologyModifier')
    # # root.addObject('Hexa2TetraTopologicalMapping', input="@grid", output="@topology", swapping="1")

    # root.addObject('SVKElasticForcefield_SOFA', youngModulus="3000", poissonRatio="0.3", topology="@topology")
    # # root.addObject('SaintVenantKirchhoffMaterial', young_modulus=3000, poisson_ratio=0.3)
    # # root.addObject('HyperelasticForcefield', printLog=True)

    # root.addObject('BoxROI', name="fixed_roi", box="-7.5 -7.5 -0.9 7.5 7.5 0.1")
    # root.addObject('FixedConstraint', indices="@fixed_roi.indices")
    # root.addObject('BoxROI', name="top_roi", box="-7.5 -7.5 79.9 7.5 7.5 80.1")
    # root.addObject('ConstantForceField', force="0 -10 0", indices="@top_roi.indices")


    return root


# Function used only if this script is called from a python environment
if __name__ == '__main__':
    main()