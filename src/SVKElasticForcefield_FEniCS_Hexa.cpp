#include "SVKElasticForcefield_FEniCS_Hexa.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>
#include "../fenics/SaintVenantKirchhoff_Hexa.h"
#include <iostream>

SVKElasticForcefield_FEniCS_Hexa::SVKElasticForcefield_FEniCS_Hexa()
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

void SVKElasticForcefield_FEniCS_Hexa::init() {
    ForceField::init();

    if (!this->mstate.get() || !d_topology_container.get()) {
        msg_error() << "Both a mechanical object and a topology container are required";
    }
}

double SVKElasticForcefield_FEniCS_Hexa::getPotentialEnergy(const sofa::core::MechanicalParams *,
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
            topology->getHexas().data()->data(), topology->getNbHexahedra(), Element::NumberOfNodes
    );

    double Psi = 0.;

    const auto nb_elements = topology->getNbHexahedra();
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

void SVKElasticForcefield_FEniCS_Hexa::addForce(const sofa::core::MechanicalParams */*mparams*/,
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
    const int geometric_dimension= form_SaintVenantKirchhoff_Hexa_F->finite_elements[0]->geometric_dimension;
    const int space_dimension= form_SaintVenantKirchhoff_Hexa_F->finite_elements[0]->space_dimension;
    const int num_element_support_dofs = form_SaintVenantKirchhoff_Hexa_F->dofmaps[0]->num_element_support_dofs;

    Eigen::Matrix <double, 1, Eigen::Dynamic, Eigen::RowMajor> F_local(1, space_dimension);
    Eigen::Matrix <double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> coefficients(num_element_support_dofs, geometric_dimension);
    const ufc_scalar_t constants[2] = {young_modulus, poisson_ratio};

    // Get the single cell integral
    const ufc_integral *integral =
        form_SaintVenantKirchhoff_Hexa_F->integrals(ufc_integral_type::cell)[0];

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
            topology->getHexas().data()->data(), topology->getNbHexahedra(), Element::NumberOfNodes
    );

    // Assemble the residual vector
    const auto nb_elements = topology->getNbHexahedra();
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
//        if(element_id==0) std::cout << F_local << "\n\n";
        for (Eigen::Index i = 0; i < Element::NumberOfNodes; ++i) {
            R.row(node_indices(element_id, i)).noalias() -= F_local.block<1,3>(0, i*3, 1, 3);
        }
    }
}

void SVKElasticForcefield_FEniCS_Hexa::addKToMatrix(sofa::defaulttype::BaseMatrix * matrix,
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
    const int geometric_dimension= form_SaintVenantKirchhoff_Hexa_J->finite_elements[0]->geometric_dimension;
    const int space_dimension= form_SaintVenantKirchhoff_Hexa_J->finite_elements[0]->space_dimension;
    const int num_element_support_dofs = form_SaintVenantKirchhoff_Hexa_J->dofmaps[0]->num_element_support_dofs;
    Eigen::Matrix <double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> K_local(space_dimension, space_dimension);
    Eigen::Matrix <double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> coefficients(num_element_support_dofs, geometric_dimension);
    const ufc_scalar_t constants[2] = {young_modulus, poisson_ratio};

    // Get the single cell integral
    const ufc_integral *integral =
        form_SaintVenantKirchhoff_Hexa_J->integrals(ufc_integral_type::cell)[0];

    const ufc_scalar_t w45[24] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const ufc_scalar_t c45[2] = {3000, 0.3};
    Eigen::Matrix<double,1, 3, Eigen::RowMajor> p_0;
    Eigen::Matrix<double,1, 3, Eigen::RowMajor> p_1;
    Eigen::Matrix<double,1, 3, Eigen::RowMajor> p_2;
    Eigen::Matrix<double,1, 3, Eigen::RowMajor> p_3;
    Eigen::Matrix<double,1, 3, Eigen::RowMajor> p_4;
    Eigen::Matrix<double,1, 3, Eigen::RowMajor> p_5;
    Eigen::Matrix<double,1, 3, Eigen::RowMajor> p_6;
    Eigen::Matrix<double,1, 3, Eigen::RowMajor> p_7;
    p_0 << 0,0,0;
    p_1 << 1,0,0;
    p_2 << 1,1,0;
    p_3 << 0,1,0;
    p_4 << 0,0,1;
    p_5 << 1,0,1;
    p_6 << 1,1,1;
    p_7 << 0,1,1;

    Eigen::Matrix<double, 8, 3, Eigen::RowMajor> cf45;
    cf45.row(0) = p_4;
    cf45.row(1) = p_5;
    cf45.row(2) = p_0;
    cf45.row(3) = p_1;
    cf45.row(4) = p_7;
    cf45.row(5) = p_6;
    cf45.row(6) = p_3;
    cf45.row(7) = p_2;




    K_local.setZero();
    // Call of the C Kernel generated by FEniCS to compute the local residual vector
    integral->tabulate_tensor(K_local.data(), w45, c45, cf45.data(), nullptr, nullptr);
    std::cout << K_local.norm();

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
            topology->getHexas().data()->data(), topology->getNbHexahedra(), Element::NumberOfNodes
    );

    // Assemble the stiffness matrix
    const auto nb_elements = topology->getNbHexahedra();
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

void SVKElasticForcefield_FEniCS_Hexa::addDForce(const sofa::core::MechanicalParams * /*mparams*/,
                                     SVKElasticForcefield_FEniCS_Hexa::Data<sofa::type::vector<sofa::type::Vec3>> & /*d_df*/,
                                     const SVKElasticForcefield_FEniCS_Hexa::Data<sofa::type::vector<sofa::type::Vec3>> & /*d_dx*/) {
    // Here you would compute df = K*dx
}

void SVKElasticForcefield_FEniCS_Hexa::draw(const sofa::core::visual::VisualParams *vparams) {
    auto * topology = d_topology_container.get();
    if (!topology)
        return;

    if (!vparams->displayFlags().getShowForceFields())
        return;

    vparams->drawTool()->saveLastState();

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0,true);

    vparams->drawTool()->disableLighting();

    const VecCoord& x = this->mstate->read(sofa::core::ConstVecCoordId::position())->getValue();

    std::vector< sofa::type::Vector3 > points[6];
    const auto number_of_elements = topology->getNbHexahedra();
    for (std::size_t hexa_id = 0; hexa_id < number_of_elements; ++hexa_id) {
        const auto & node_indices = topology->getHexahedron(static_cast<sofa::Index>(hexa_id));

        auto a = node_indices[0];
        auto b = node_indices[1];
        auto d = node_indices[3];
        auto c = node_indices[2];
        auto e = node_indices[4];
        auto f = node_indices[5];
        auto h = node_indices[7];
        auto g = node_indices[6];


        Coord center = (x[a]+x[b]+x[c]+x[d]+x[e]+x[g]+x[f]+x[h])*0.125;
        Real percentage = 0.15;
        Coord pa = x[a]-(x[a]-center)*percentage;
        Coord pb = x[b]-(x[b]-center)*percentage;
        Coord pc = x[c]-(x[c]-center)*percentage;
        Coord pd = x[d]-(x[d]-center)*percentage;
        Coord pe = x[e]-(x[e]-center)*percentage;
        Coord pf = x[f]-(x[f]-center)*percentage;
        Coord pg = x[g]-(x[g]-center)*percentage;
        Coord ph = x[h]-(x[h]-center)*percentage;



        points[0].push_back(pa);
        points[0].push_back(pb);
        points[0].push_back(pc);
        points[0].push_back(pa);
        points[0].push_back(pc);
        points[0].push_back(pd);

        points[1].push_back(pe);
        points[1].push_back(pf);
        points[1].push_back(pg);
        points[1].push_back(pe);
        points[1].push_back(pg);
        points[1].push_back(ph);

        points[2].push_back(pc);
        points[2].push_back(pd);
        points[2].push_back(ph);
        points[2].push_back(pc);
        points[2].push_back(ph);
        points[2].push_back(pg);

        points[3].push_back(pa);
        points[3].push_back(pb);
        points[3].push_back(pf);
        points[3].push_back(pa);
        points[3].push_back(pf);
        points[3].push_back(pe);

        points[4].push_back(pa);
        points[4].push_back(pd);
        points[4].push_back(ph);
        points[4].push_back(pa);
        points[4].push_back(ph);
        points[4].push_back(pe);

        points[5].push_back(pb);
        points[5].push_back(pc);
        points[5].push_back(pg);
        points[5].push_back(pb);
        points[5].push_back(pg);
        points[5].push_back(pf);
    }

    vparams->drawTool()->drawTriangles(points[0], sofa::type::RGBAColor(0.7f,0.7f,0.1f,1.0f));
    vparams->drawTool()->drawTriangles(points[1], sofa::type::RGBAColor(0.7f,0.0f,0.0f,1.0f));
    vparams->drawTool()->drawTriangles(points[2], sofa::type::RGBAColor(0.0f,0.7f,0.0f,1.0f));
    vparams->drawTool()->drawTriangles(points[3], sofa::type::RGBAColor(0.0f,0.0f,0.7f,1.0f));
    vparams->drawTool()->drawTriangles(points[4], sofa::type::RGBAColor(0.1f,0.7f,0.7f,1.0f));
    vparams->drawTool()->drawTriangles(points[5], sofa::type::RGBAColor(0.7f,0.1f,0.7f,1.0f));


    std::vector< sofa::type::Vector3 > ignored_points[6];
    for (std::size_t hexa_id = 0; hexa_id < number_of_elements; ++hexa_id) {
        const auto & node_indices = topology->getHexahedron(static_cast<sofa::Index>(hexa_id));

        auto a = node_indices[0];
        auto b = node_indices[1];
        auto d = node_indices[3];
        auto c = node_indices[2];
        auto e = node_indices[4];
        auto f = node_indices[5];
        auto h = node_indices[7];
        auto g = node_indices[6];


        Coord center = (x[a]+x[b]+x[c]+x[d]+x[e]+x[g]+x[f]+x[h])*0.125;
        Real percentage = 0.15;
        Coord pa = x[a]-(x[a]-center)*percentage;
        Coord pb = x[b]-(x[b]-center)*percentage;
        Coord pc = x[c]-(x[c]-center)*percentage;
        Coord pd = x[d]-(x[d]-center)*percentage;
        Coord pe = x[e]-(x[e]-center)*percentage;
        Coord pf = x[f]-(x[f]-center)*percentage;
        Coord pg = x[g]-(x[g]-center)*percentage;
        Coord ph = x[h]-(x[h]-center)*percentage;



        ignored_points[0].push_back(pa);
        ignored_points[0].push_back(pb);
        ignored_points[0].push_back(pc);
        ignored_points[0].push_back(pa);
        ignored_points[0].push_back(pc);
        ignored_points[0].push_back(pd);

        ignored_points[1].push_back(pe);
        ignored_points[1].push_back(pf);
        ignored_points[1].push_back(pg);
        ignored_points[1].push_back(pe);
        ignored_points[1].push_back(pg);
        ignored_points[1].push_back(ph);

        ignored_points[2].push_back(pc);
        ignored_points[2].push_back(pd);
        ignored_points[2].push_back(ph);
        ignored_points[2].push_back(pc);
        ignored_points[2].push_back(ph);
        ignored_points[2].push_back(pg);

        ignored_points[3].push_back(pa);
        ignored_points[3].push_back(pb);
        ignored_points[3].push_back(pf);
        ignored_points[3].push_back(pa);
        ignored_points[3].push_back(pf);
        ignored_points[3].push_back(pe);

        ignored_points[4].push_back(pa);
        ignored_points[4].push_back(pd);
        ignored_points[4].push_back(ph);
        ignored_points[4].push_back(pa);
        ignored_points[4].push_back(ph);
        ignored_points[4].push_back(pe);

        ignored_points[5].push_back(pb);
        ignored_points[5].push_back(pc);
        ignored_points[5].push_back(pg);
        ignored_points[5].push_back(pb);
        ignored_points[5].push_back(pg);
        ignored_points[5].push_back(pf);
    }

    vparams->drawTool()->drawTriangles(ignored_points[0], sofa::type::RGBAColor(0.49f,0.49f,0.49f,0.3f));
    vparams->drawTool()->drawTriangles(ignored_points[1], sofa::type::RGBAColor(0.49f,0.49f,0.49f,0.3f));
    vparams->drawTool()->drawTriangles(ignored_points[2], sofa::type::RGBAColor(0.49f,0.49f,0.49f,0.3f));
    vparams->drawTool()->drawTriangles(ignored_points[3], sofa::type::RGBAColor(0.49f,0.49f,0.49f,0.3f));
    vparams->drawTool()->drawTriangles(ignored_points[4], sofa::type::RGBAColor(0.49f,0.49f,0.49f,0.3f));
    vparams->drawTool()->drawTriangles(ignored_points[5], sofa::type::RGBAColor(0.49f,0.49f,0.49f,0.3f));


    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0,false);

    vparams->drawTool()->restoreLastState();
}

void SVKElasticForcefield_FEniCS_Hexa::computeBBox(const sofa::core::ExecParams * /*params*/, bool onlyVisible) {
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
 .add<SVKElasticForcefield_FEniCS_Hexa>();
