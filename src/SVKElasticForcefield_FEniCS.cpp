#include "SVKElasticForcefield_FEniCS.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>
#include "../fenics/SaintVenantKirchhoff_Tetra.h"
#include <iostream>

SVKElasticForcefield_FEniCS::SVKElasticForcefield_FEniCS()
: d_youngModulus(initData(&d_youngModulus,
                          Real(1000), "youngModulus",
                          "Young's modulus of the material",
                          true /*displayed_in_GUI*/, false /*read_only_in_GUI*/))
, d_poissonRatio(initData(&d_poissonRatio,
                          Real(0.3),  "poissonRatio",
                          "Poisson's ratio of the material",
                          true /*displayed_in_GUI*/, false /*read_only_in_GUI*/))
, d_topology_container(initLink(
        "topology", "Topology that contains the elements on which this force will be computed."))
{
}

void SVKElasticForcefield_FEniCS::init() {
    ForceField::init();

    if (!this->mstate.get() || !d_topology_container.get()) {
        msg_error() << "Both a mechanical object and a topology container are required";
    }
}

double SVKElasticForcefield_FEniCS::getPotentialEnergy(const sofa::core::MechanicalParams *,
                                                              const Data<sofa::type::vector<Coord>> & d_x) const {
    using Mat33 = Eigen::Matrix<double, 3, 3>;


    if (!this->mstate.get() || !d_topology_container.get()) {
        return 0;
    }

    auto * state = this->mstate.get();
    auto * topology = d_topology_container.get();
    const auto  poisson_ratio = d_poissonRatio.getValue();
    const auto  young_modulus = d_youngModulus.getValue();
    const auto mu = young_modulus / (2.0 * (1.0 + poisson_ratio));
    const auto lm = young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
    static const auto Id = Mat33::Identity();


    // Convert SOFA input position vector to an Eigen matrix (nx3 for n nodes)
    auto sofa_x = sofa::helper::getReadAccessor(d_x);
    Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>>  x (sofa_x.ref().data()->data(),  state->getSize(), 3);

    // Convert the node index vector from SOFA to an Eigen matrix (nxm for n elements of m nodes each)
    Eigen::Map<const Eigen::Matrix<sofa::Index, Eigen::Dynamic, Element::NumberOfNodes, Eigen::RowMajor>> node_indices (
            topology->getTetras().data()->data(), topology->getNbTetrahedra(), Element::NumberOfNodes
    );

    double Psi = 0.;

    const auto nb_elements = topology->getNbTetrahedra();
    for (Eigen::Index element_id = 0; element_id < nb_elements; ++element_id) {
        // Position vector of each of the element nodes
        Eigen::Matrix<double, Element::NumberOfNodes, 3, Eigen::RowMajor> node_positions;
        for (Eigen::Index node_id = 0; node_id < Element::NumberOfNodes; ++node_id) {
            node_positions.row(node_id) = x.row(node_indices(element_id, node_id));
        }

        for (const GaussNode & gauss_node : p_gauss_nodes[element_id]) {
            // Jacobian of the gauss node's transformation mapping from the elementary space to the world space
            const auto & detJ = gauss_node.jacobian_determinant;

            // Derivatives of the shape functions at the gauss node with respect to global coordinates x,y and z
            const auto & dN_dx = gauss_node.dN_dx;

            // Gauss quadrature node weight
            const auto & w = gauss_node.weight;

            // Deformation tensor at gauss node
            const Mat33 F = node_positions.transpose()*dN_dx;

            // Green-Lagrange strain tensor at gauss node
            const Mat33 E = 1/2. * (F.transpose() * F - Id);
            const double trE  = E.trace();
            const double trEE = (E*E).trace();

            // Add the potential energy at gauss node
            Psi += (detJ * w) * lm/2.*(trE*trE) + mu*trEE;
        }
    }

    return Psi;
}

void SVKElasticForcefield_FEniCS::addForce(const sofa::core::MechanicalParams */*mparams*/,
                                                  Data<sofa::type::vector<Deriv>> &d_f,
                                                  const Data<sofa::type::vector<Coord>> &d_x,
                                                  const Data<sofa::type::vector<Deriv>> &/*d_v*/) {
    using namespace sofa::helper::logging;

    if (!this->mstate.get() || !d_topology_container.get()) {
        return;
    }

    auto * state = this->mstate.get();
    auto * topology = d_topology_container.get();
    const auto  poisson_ratio = d_poissonRatio.getValue();
    const auto  young_modulus = d_youngModulus.getValue();


    //    FEniCs variables
    const int geometric_dimension= form_SaintVenantKirchhoff_Tetra_F->finite_elements[0]->geometric_dimension;
    const int space_dimension= form_SaintVenantKirchhoff_Tetra_F->finite_elements[0]->space_dimension;
    const int num_element_support_dofs = form_SaintVenantKirchhoff_Tetra_F->dofmaps[0]->num_element_support_dofs;

    Eigen::Matrix <double, 1, Eigen::Dynamic, Eigen::RowMajor> F_local(1, space_dimension);
    Eigen::Matrix <double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> coefficients(num_element_support_dofs, geometric_dimension);
    const ufc_scalar_t constants[2] = {young_modulus, poisson_ratio};

    // Get the single cell integral
    const ufc_integral *integral =
        form_SaintVenantKirchhoff_Tetra_F->integrals(ufc_integral_type::cell)[0];


    // Convert SOFA input rest position vector to an Eigen matrix (nx3 for n nodes)
    auto sofa_x0 = this->mstate->readRestPositions();
    const Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>>    x0      (sofa_x0.ref().data()->data(), state->getSize(), 3);

    // Convert SOFA input position vector to an Eigen matrix (nx3 for n nodes)
    auto sofa_x = sofa::helper::getReadAccessor(d_x);
    Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>>  x (sofa_x.ref().data()->data(),  state->getSize(), 3);

    // Compute the displacement with respect to the rest position
    const auto u =  x - x0;

    // Convert SOFA output residual vector to an Eigen matrix (nx3 for n nodes)
    auto sofa_f = sofa::helper::getWriteAccessor(d_f);
    Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>>  R (&(sofa_f[0][0]),  state->getSize(), 3);

    // Convert the node index vector from SOFA to an Eigen matrix (nxm for n elements of m nodes each)
    Eigen::Map<const Eigen::Matrix<sofa::Index, Eigen::Dynamic, Element::NumberOfNodes, Eigen::RowMajor>> node_indices (
            topology->getTetras().data()->data(), topology->getNbTetrahedra(), Element::NumberOfNodes
    );

    // Assemble the residual vector
    const auto nb_elements = topology->getNbTetrahedra();
    for (Eigen::Index element_id = 0; element_id < nb_elements; ++element_id) {

        // Position vector of each of the element nodes
        Eigen::Matrix<double, Element::NumberOfNodes, 3, Eigen::RowMajor> node_positions;
        coefficients.setZero();
        for (Eigen::Index node_id = 0; node_id < Element::NumberOfNodes; ++node_id) {
            node_positions.row(node_id) = x0.row(node_indices(element_id, node_id));
            coefficients.row(node_id) = u.row(node_indices(element_id, node_id));
        }
        F_local.setZero();
        // Call of the C Kernel generated by FEniCS to compute the local residual vector
        integral->tabulate_tensor(F_local.data(), coefficients.data(), constants, node_positions.data(), nullptr, nullptr);
        for (Eigen::Index i = 0; i < Element::NumberOfNodes; ++i) {
            R.row(node_indices(element_id, i)).noalias() -= F_local.block<1,3>(0, i*3, 1, 3);
        }
    }
}

void SVKElasticForcefield_FEniCS::addKToMatrix(sofa::defaulttype::BaseMatrix * matrix,
                                                      double kFact,
                                                      unsigned int & offset) {

    if (!this->mstate.get() || !d_topology_container.get()) {
        return;
    }

    auto * state = this->mstate.get();
    auto * topology = d_topology_container.get();
    const auto  poisson_ratio = d_poissonRatio.getValue();
    const auto  young_modulus = d_youngModulus.getValue();

    //    FEniCs variables
    const int geometric_dimension= form_SaintVenantKirchhoff_Tetra_J->finite_elements[0]->geometric_dimension;
    const int space_dimension= form_SaintVenantKirchhoff_Tetra_J->finite_elements[0]->space_dimension;
    const int num_element_support_dofs = form_SaintVenantKirchhoff_Tetra_J->dofmaps[0]->num_element_support_dofs;
    Eigen::Matrix <double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> K_local(space_dimension, space_dimension);
    Eigen::Matrix <double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> coefficients(num_element_support_dofs, geometric_dimension);
    const ufc_scalar_t constants[2] = {young_modulus, poisson_ratio};

    // Get the single cell integral
    const ufc_integral *integral =
        form_SaintVenantKirchhoff_Tetra_J->integrals(ufc_integral_type::cell)[0];

    const ufc_scalar_t w45[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const ufc_scalar_t c45[2] = {3000, 0.3};
    const double cf45[12] = {-5.625, -7.5, 0, -7.5, -7.5, 4, -7.5, -5.625, 4, -7.5, -7.5, 0};

    K_local.setZero();
    // Call of the C Kernel generated by FEniCS to compute the local residual vector
    integral->tabulate_tensor(K_local.data(), w45, c45, cf45, nullptr, nullptr);
    std::cout << K_local;

    // Convert SOFA input rest position vector to an Eigen matrix (nx3 for n nodes)
    auto sofa_x0 = this->mstate->readRestPositions();
    const Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>>    x0      (sofa_x0.ref().data()->data(), state->getSize(), 3);

    // Convert SOFA input position vector to an Eigen matrix (nx3 for n nodes)
    auto sofa_x = state->readPositions();
    Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>>  x (sofa_x.ref().data()->data(),  state->getSize(), 3);

    // Compute the displacement with respect to the rest position
    const auto u =  x - x0;

    // Convert the node index vector from SOFA to an Eigen matrix (nxm for n elements of m nodes each)
    Eigen::Map<const Eigen::Matrix<sofa::Index, Eigen::Dynamic, Element::NumberOfNodes, Eigen::RowMajor>> node_indices (
            topology->getTetras().data()->data(), topology->getNbTetrahedra(), Element::NumberOfNodes
    );

    // Assemble the stiffness matrix
    const auto nb_elements = topology->getNbTetrahedra();
    for (Eigen::Index element_id = 0; element_id < nb_elements; ++element_id) {

        // Position vector of each of the element nodes
        Eigen::Matrix<double, Element::NumberOfNodes, 3, Eigen::RowMajor> node_positions;
        coefficients.setZero();
        for (Eigen::Index node_id = 0; node_id < Element::NumberOfNodes; ++node_id) {
            node_positions.row(node_id) = x0.row(node_indices(element_id, node_id));
            coefficients.row(node_id) = u.row(node_indices(element_id, node_id));
            }
        K_local.setZero();
        // Call of the C Kernel generated by FEniCS to compute the local stiffness matrix
        integral->tabulate_tensor(K_local.data(), coefficients.data(), constants, node_positions.data(), nullptr, nullptr);

        for (Eigen::Index i = 0; i < Element::NumberOfNodes; ++i) {
            const auto I   = static_cast<int>(offset + node_indices(element_id, i) * 3);
            for (int m = 0; m < 3; ++m) {
                matrix->add(I + m, I + m, K_local(3*i + m, 3*i + m));
                for (int n = m+1; n < 3; ++n) {
                    matrix->add(I + m, I + n, K_local(3*i + m, 3*i + n));
                    matrix->add(I + n, I + m, K_local(3*i + m, 3*i + n));
                }
            }
            for (Eigen::Index j = i+1; j < Element::NumberOfNodes; ++j) {
                const auto J   = static_cast<int>(offset + node_indices(element_id, j) * 3);
                for (int m = 0; m < 3; ++m) {
                    for (int n = 0; n < 3; ++n) {
                        matrix->add(I + m, J + n, K_local(3*i + m, 3*j + n));
                        matrix->add(J + n, I + m, K_local(3*i + m, 3*j + n));
                    }
                }
            }
        }
    }
}

void SVKElasticForcefield_FEniCS::addDForce(const sofa::core::MechanicalParams * /*mparams*/,
                                     SVKElasticForcefield_FEniCS::Data<sofa::type::vector<sofa::type::Vec3>> & /*d_df*/,
                                     const SVKElasticForcefield_FEniCS::Data<sofa::type::vector<sofa::type::Vec3>> & /*d_dx*/) {
    // Here you would compute df = K*dx
}

void SVKElasticForcefield_FEniCS::draw(const sofa::core::visual::VisualParams *vparams) {
    if (!this->mstate.get() || !d_topology_container.get()) {
        return;
    }

    if (!vparams->displayFlags().getShowForceFields()) {
        return;
    }

    auto * state = this->mstate.get();
    auto * topology = d_topology_container.get();

    vparams->drawTool()->disableLighting();
    const auto x = state->readPositions();

    std::vector< sofa::type::Vec<3, double> > points[4];
    const auto number_of_elements = topology->getNbTetrahedra();
    for (sofa::core::topology::Topology::TetrahedronID i = 0 ; i<number_of_elements;++i) {
        const auto t=topology->getTetra(i);

        const auto & a = t[0];
        const auto & b = t[1];
        const auto & c = t[2];
        const auto & d = t[3];
        Coord center = (x[a]+x[b]+x[c]+x[d])*0.125;
        Coord pa = (x[a]+center)*(Real)0.666667;
        Coord pb = (x[b]+center)*(Real)0.666667;
        Coord pc = (x[c]+center)*(Real)0.666667;
        Coord pd = (x[d]+center)*(Real)0.666667;

        points[0].push_back(pa);
        points[0].push_back(pb);
        points[0].push_back(pc);

        points[1].push_back(pb);
        points[1].push_back(pc);
        points[1].push_back(pd);

        points[2].push_back(pc);
        points[2].push_back(pd);
        points[2].push_back(pa);

        points[3].push_back(pd);
        points[3].push_back(pa);
        points[3].push_back(pb);
    }

    sofa::type::RGBAColor face_colors[4] = {
            {1.0, 0.0, 0.0, 1.0},
            {1.0, 0.0, 0.5, 1.0},
            {1.0, 1.0, 0.0, 1.0},
            {1.0, 0.5, 1.0, 1.0}
    };

    vparams->drawTool()->drawTriangles(points[0], face_colors[0]);
    vparams->drawTool()->drawTriangles(points[1], face_colors[1]);
    vparams->drawTool()->drawTriangles(points[2], face_colors[2]);
    vparams->drawTool()->drawTriangles(points[3], face_colors[3]);

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0,false);

    vparams->drawTool()->restoreLastState();
}

void SVKElasticForcefield_FEniCS::computeBBox(const sofa::core::ExecParams * /*params*/, bool onlyVisible) {
    using namespace sofa::core::objectmodel;

    if (!onlyVisible) return;
    if (!this->mstate) return;

    sofa::helper::ReadAccessor<Data < VecCoord>>
            x = this->mstate->read(sofa::core::VecCoordId::position());

    static const Real max_real = std::numeric_limits<Real>::max();
    static const Real min_real = std::numeric_limits<Real>::lowest();
    Real maxBBox[3] = {min_real, min_real, min_real};
    Real minBBox[3] = {max_real, max_real, max_real};
    for (size_t i = 0; i < x.size(); i++) {
        for (int c = 0; c < 3; c++) {
            if (x[i][c] > maxBBox[c]) maxBBox[c] = static_cast<Real>(x[i][c]);
            else if (x[i][c] < minBBox[c]) minBBox[c] = static_cast<Real>(x[i][c]);
        }
    }

    this->f_bbox.setValue(sofa::type::TBoundingBox<Real>(minBBox, maxBBox));
}


using sofa::core::RegisterObject;
[[maybe_unused]]
static int _c_ = RegisterObject("Simple implementation of a Saint-Venant-Kirchhoff force field for tetrahedral meshes.")
 .add<SVKElasticForcefield_FEniCS>();
