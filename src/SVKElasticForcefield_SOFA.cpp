#include "SVKElasticForcefield_SOFA.h"
#include <sofa/version.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/simulation/Node.h>
#include <sofa/helper/AdvancedTimer.h>
#include <iostream>

SVKElasticForcefield_SOFA::SVKElasticForcefield_SOFA()
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
void SVKElasticForcefield_SOFA::init()
{
    using sofa::core::topology::BaseMeshTopology;
    using sofa::core::objectmodel::BaseContext;
    using CaribouTopology = SofaCaribou::topology::CaribouTopology<Tetrahedron>;

    Inherit1::init();

    auto *context = this->getContext();

    if (!this->mstate) {
        msg_warning() << "No mechanical object found in the current context node. The data parameter "
                      << "'" << this->mstate.getName() << "' can be use to set the path to a mechanical "
                      << "object having a template of '" << DataTypes::Name() << "'";
        return;
    }

    // If not topology is specified, try to find one automatically in the current context
    if (not d_topology_container.get()) {
        // No topology specified. Try to find one suitable.
        auto caribou_containers = context->template getObjects<CaribouTopology>(
                BaseContext::Local);
        auto sofa_containers = context->template getObjects<BaseMeshTopology>(BaseContext::Local);
        std::vector<BaseMeshTopology *> sofa_compatible_containers;
        for (auto container : sofa_containers) {
            if (CaribouTopology::mesh_is_compatible(container)) {
                sofa_compatible_containers.push_back(container);
            }
        }
        if (caribou_containers.empty() and sofa_compatible_containers.empty()) {
            msg_warning() << "Could not find a topology container in the current context. "
                          << "Please add a compatible one in the current context or set the "
                          << "container's path using the '" << d_topology_container.getName()
                          << "' data parameter.";
        } else {
            if (caribou_containers.size() + sofa_compatible_containers.size() > 1) {
                msg_warning() << "Multiple topologies were found in the context node. "
                              << "Please specify which one contains the elements on "
                              << "which this force field will be applied "
                              << "by explicitly setting the container's path in the  '"
                              << d_topology_container.getName() << "' data parameter.";
            } else {
                // Prefer caribou's containers first
                if (not caribou_containers.empty()) {
                    d_topology_container.set(caribou_containers[0]);
                } else {
                    d_topology_container.set(sofa_compatible_containers[0]);
                }
            }

            msg_info() << "Automatically found the topology '" << d_topology_container.get()->getPathName()
                       << "'.";
        }
    }

    // Create a caribou internal Domain over the topology
    if (d_topology_container.get()) {
        auto sofa_topology = dynamic_cast<BaseMeshTopology *>(d_topology_container.get());
        auto caribou_topology = dynamic_cast<CaribouTopology *>(d_topology_container.get());
        if (sofa_topology) {
            // Initialize a new caribou topology from the SOFA topology
            p_topology = sofa::core::objectmodel::New<CaribouTopology>();
//            p_topology->findData("indices")->setParent(CaribouTopology::get_indices_data_from(sofa_topology));
            p_topology->findData("indices")->setParent(sofa_topology->findData("tetrahedra"));
            p_topology->findData("position")->setParent(this->getMState()->findData("position"));
            p_topology->init();
        } else {
            // A Caribou topology already exists in the scene
            p_topology = caribou_topology;
        }

        if (number_of_elements() == 0) {
            msg_warning() << "No element found in the topology '" << d_topology_container.get()->getPathName() << "'";
        }
    }

    using namespace sofa::core::objectmodel;

    if (!this->mstate)
        return;

    // Resize the container of elements'quadrature nodes
    const auto nb_elements = this->number_of_elements();
    if (p_elements_quadrature_nodes.size() != nb_elements) {
        p_elements_quadrature_nodes.resize(nb_elements);
    }

    // Translate the Sofa's mechanical state vector to Eigen vector type
    sofa::helper::ReadAccessor<Data<VecCoord>> sofa_x0 = this->mstate->readRestPositions();
    const Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, Dimension, Eigen::RowMajor>>    X0      (sofa_x0.ref().data()->data(), sofa_x0.size(), Dimension);

    // Loop on each element and compute the shape functions and their derivatives for every of their integration points
    for (std::size_t element_id = 0; element_id < nb_elements; ++element_id) {

        // Get an Element instance from the Domain
        const auto initial_element = this->topology()->element(element_id);

        // Fill in the Gauss integration nodes for this element
        p_elements_quadrature_nodes[element_id] = get_gauss_nodes(element_id, initial_element);
    }

    // Compute the volume
    Real v = 0.;
    for (std::size_t element_id = 0; element_id < nb_elements; ++element_id) {
        for (const auto & gauss_node : gauss_nodes_of(element_id)) {
            // Jacobian of the gauss node's transformation mapping from the elementary space to the world space
            const auto detJ = gauss_node.jacobian_determinant;

            // Gauss quadrature node weight
            const auto w = gauss_node.weight;

            v += detJ*w;
        }
    }
    msg_info() << "Total volume of the geometry is " << v;
}

auto SVKElasticForcefield_SOFA::get_gauss_nodes(const std::size_t & /*element_id*/, const Tetrahedron & element) const -> GaussContainer {
    GaussContainer gauss_nodes {};

    const auto nb_of_gauss_nodes = gauss_nodes.size();
    for (std::size_t gauss_node_id = 0; gauss_node_id < nb_of_gauss_nodes; ++gauss_node_id) {
        const auto & g = element.gauss_node(gauss_node_id);

        const auto J = element.jacobian(g.position);
        const Mat33 Jinv = J.inverse();
        const auto detJ = std::abs(J.determinant());

        // Derivatives of the shape functions at the gauss node with respect to global coordinates x,y and z
        const Matrix<NumberOfNodesPerElement, Dimension> dN_dx =
            (Jinv.transpose() * element.dL(g.position).transpose()).transpose();


        GaussNode & gauss_node = gauss_nodes[gauss_node_id];
        gauss_node.weight               = g.weight;
        gauss_node.jacobian_determinant = detJ;
        gauss_node.dN_dx                = dN_dx;
    }

    return gauss_nodes;
}

double SVKElasticForcefield_SOFA::getPotentialEnergy(const sofa::core::MechanicalParams *,
                                                              const Data<sofa::type::vector<Coord>> & d_x) const {
    using Mat33 = Eigen::Matrix<double, 3, 3>;

    if (!this->mstate.get() || !d_topology_container.get()) {
        return 0;
    }

    auto * state = this->mstate.get();
    const auto  poisson_ratio = d_poissonRatio.getValue();
    const auto  young_modulus = d_youngModulus.getValue();
    const auto mu = young_modulus / (2.0 * (1.0 + poisson_ratio));
    const auto lm = young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
    static const auto Id = Mat33::Identity();


    // Convert SOFA input position vector to an Eigen matrix (nx3 for n nodes)
    auto sofa_x = sofa::helper::getReadAccessor(d_x);
    Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>>  x (sofa_x.ref().data()->data(),  state->getSize(), 3);

    double Psi = 0.;

    const auto nb_elements = this->number_of_elements();

    for (std::size_t element_id = 0; element_id < nb_elements; ++element_id) {
        // Fetch the node indices of the element
        auto node_indices = this->topology()->domain()->element_indices(element_id);

        // Fetch the initial and current positions of the element's nodes
        Matrix<NumberOfNodesPerElement, Dimension> node_positions;

        for (std::size_t i = 0; i < NumberOfNodesPerElement; ++i) {
            node_positions.row(i).noalias() = x.row(node_indices[i]);
        }

        for (const GaussNode & gauss_node : p_elements_quadrature_nodes[element_id]) {

            // Jacobian of the gauss node's transformation mapping from the elementary space to the world space
            const auto & detJ = gauss_node.jacobian_determinant;

            // Derivatives of the shape functions at the gauss node with respect to global coordinates x,y and z
            const auto & dN_dx = gauss_node.dN_dx;

            // Gauss quadrature node weight
            const auto & w = gauss_node.weight;

            // Deformation tensor at gauss node
            const auto & F = node_positions.transpose()*dN_dx;

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

void SVKElasticForcefield_SOFA::addForce(const sofa::core::MechanicalParams */*mparams*/,
                                                  Data<sofa::type::vector<Deriv>> &d_f,
                                                  const Data<sofa::type::vector<Coord>> &d_x,
                                                  const Data<sofa::type::vector<Deriv>> &/*d_v*/) {
    using Mat33 = Eigen::Matrix<double, 3, 3>;
    using namespace sofa::helper;


    if (!this->mstate.get() || !d_topology_container.get()) {
        return;
    }

    const auto  poisson_ratio = d_poissonRatio.getValue();
    const auto  young_modulus = d_youngModulus.getValue();
    const auto mu = young_modulus / (2.0 * (1.0 + poisson_ratio));
    const auto lm = young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
    static const auto Id = Mat33::Identity();

    ReadAccessor<Data<VecCoord>> sofa_x = d_x;
    WriteAccessor<Data<VecDeriv>> sofa_f = d_f;

    if (sofa_x.size() != sofa_f.size())
        return;
    const auto nb_nodes = sofa_x.size();
    const auto nb_elements = this->number_of_elements();

    if (nb_nodes == 0 || nb_elements == 0)
        return;

    if (p_elements_quadrature_nodes.size() != nb_elements)
        return;

    Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, Dimension, Eigen::RowMajor>>    X       (sofa_x.ref().data()->data(),  nb_nodes, Dimension);
    Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, Dimension, Eigen::RowMajor>> forces  (&(sofa_f[0][0]),  nb_nodes, Dimension);

    sofa::helper::AdvancedTimer::stepBegin("HyperelasticForcefield::addForce");

    for (std::size_t element_id = 0; element_id < nb_elements; ++element_id) {

        // Fetch the node indices of the element
        auto node_indices = this->topology()->domain()->element_indices(element_id);

        // Fetch the initial and current positions of the element's nodes
        Matrix<NumberOfNodesPerElement, Dimension> current_nodes_position;

        for (std::size_t i = 0; i < NumberOfNodesPerElement; ++i) {
            current_nodes_position.row(i).noalias() = X.row(node_indices[i]);
        }

        // Compute the nodal forces
        Matrix<NumberOfNodesPerElement, Dimension> nodal_forces;
        nodal_forces.fill(0);

        for (GaussNode &gauss_node : p_elements_quadrature_nodes[element_id]) {

            // Jacobian of the gauss node's transformation mapping from the elementary space to the world space
            const auto & detJ = gauss_node.jacobian_determinant;

            // Derivatives of the shape functions at the gauss node with respect to global coordinates x,y and z
            const auto & dN_dx = gauss_node.dN_dx;

            // Gauss quadrature node weight
            const auto & w = gauss_node.weight;

            // Deformation tensor at gauss node
            const Mat33 F = current_nodes_position.transpose()*dN_dx;

            // Right Cauchy-Green strain tensor at gauss node
            const Mat33 E = 1/2. * (F.transpose() * F - Id);

            // Second Piola-Kirchhoff stress tensor at gauss node
            const Mat33 S = lm*E.trace()*Id + 2*mu*E;

            // Elastic forces w.r.t the gauss node applied on each nodes
            for (size_t i = 0; i < NumberOfNodesPerElement; ++i) {
                const auto dx = dN_dx.row(i).transpose();
                const Vector<Dimension> f_ = (detJ * w) * F*S*dx;
                for (size_t j = 0; j < Dimension; ++j) {
                    nodal_forces(i, j) += f_[j];
                }
            }
        }

        for (size_t i = 0; i < NumberOfNodesPerElement; ++i) {
            for (size_t j = 0; j < Dimension; ++j) {
                sofa_f[node_indices[i]][j] -= nodal_forces(i,j);
            }
        }
    }
}

void SVKElasticForcefield_SOFA::addKToMatrix(sofa::defaulttype::BaseMatrix * matrix,
                                                      double kFact,
                                                      unsigned int & offset) {
    assemble_stiffness();

    if (!this->mstate.get() || !d_topology_container.get()) {
        return;
    }

    for (int k = 0; k < p_K.outerSize(); ++k) {
        for (typename Eigen::SparseMatrix<Real>::InnerIterator it(p_K, k); it; ++it) {
            const auto i = it.row();
            const auto j = it.col();
            const auto v = -1 * it.value() * kFact;
            if (i != j) {
                matrix->add(offset+i, offset+j, v);
                matrix->add(offset+j, offset+i, v);
            } else {
                matrix->add(offset+i, offset+i, v);
            }
        }
    }
}

void SVKElasticForcefield_SOFA::assemble_stiffness()
{
    assemble_stiffness(*this->mstate->read (p_X_id.getId(this->mstate)));
}

void SVKElasticForcefield_SOFA::assemble_stiffness(const sofa::core::objectmodel::Data<VecCoord> & x) {
    using namespace sofa::core::objectmodel;

    const sofa::helper::ReadAccessor<Data<VecCoord>> sofa_x= x;
    const auto nb_nodes = sofa_x.size();
    Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, Dimension, Eigen::RowMajor>>    X       (sofa_x.ref().data()->data(),  nb_nodes, Dimension);

    assemble_stiffness(X);
}

template<typename Derived>
void SVKElasticForcefield_SOFA::assemble_stiffness(const Eigen::MatrixBase<Derived> & x) {

    const auto  poisson_ratio = d_poissonRatio.getValue();
    const auto  young_modulus = d_youngModulus.getValue();
    const auto mu = young_modulus / (2.0 * (1.0 + poisson_ratio));
    const auto lm = young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
    static const auto Id = Mat33::Identity();
    const auto nb_elements = this->number_of_elements();
    const auto nb_nodes = x.rows();
    const auto nDofs = nb_nodes*Dimension;
    p_K.resize(nDofs, nDofs);

    ///< Triplets are used to store matrix entries before the call to 'compress'.
    /// Duplicates entries are summed up.
    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(nDofs*24*2);

    for (int element_id = 0; element_id < static_cast<int>(nb_elements); ++element_id) {
        // Fetch the node indices of the element
        auto node_indices = this->topology()->domain()->element_indices(element_id);

        // Fetch the current positions of the element's nodes
        Matrix<NumberOfNodesPerElement, Dimension> current_nodes_position;

        for (std::size_t i = 0; i < NumberOfNodesPerElement; ++i) {
            current_nodes_position.row(i).noalias() = x.row(node_indices[i]).template cast<Real>();
        }

        using Stiffness = Eigen::Matrix<FLOATING_POINT_TYPE, NumberOfNodesPerElement*Dimension, NumberOfNodesPerElement*Dimension, Eigen::RowMajor>;
        Stiffness Ke = Stiffness::Zero();

        for (const auto & gauss_node : gauss_nodes_of(element_id)) {
            // Jacobian of the gauss node's transformation mapping from the elementary space to the world space
            const auto detJ = gauss_node.jacobian_determinant;

            // Derivatives of the shape functions at the gauss node with respect to global coordinates x,y and z
            const auto dN_dx = gauss_node.dN_dx;

            // Gauss quadrature node weight
            const auto w = gauss_node.weight;

            // Deformation tensor at gauss node
            const Mat33 F = current_nodes_position.transpose()*dN_dx;

            // Green-Lagrange strain tensor at gauss node
            const Mat33 E = 1/2. * (F.transpose() * F - Id);

            // Second Piola-Kirchhoff stress tensor at gauss node
            const Mat33 S = lm*E.trace()*Id + 2*mu*E;

            Eigen::Matrix<double, 6, 6> D;
            D <<
                   lm + 2*mu,  lm,         lm,       0,  0,  0,
                   lm,      lm + 2*mu,     lm,       0,  0,  0,
                   lm,         lm,       lm + 2*mu,  0,  0,  0,
                    0,          0,          0,      mu,  0,  0,
                    0,          0,          0,       0, mu,  0,
                    0,          0,          0,       0,  0, mu;

            // Computation of the tangent-stiffness matrix
            for (std::size_t i = 0; i < NumberOfNodesPerElement; ++i) {
                // Derivatives of the ith shape function at the gauss node with respect to global coordinates x,y and z
                const Vector<3> dxi = dN_dx.row(i).transpose();

                Matrix<6,3> Bi;
                Bi <<
                   F(0,0)*dxi[0],                 F(1,0)*dxi[0],                 F(2,0)*dxi[0],
                        F(0,1)*dxi[1],                 F(1,1)*dxi[1],                 F(2,1)*dxi[1],
                        F(0,2)*dxi[2],                 F(1,2)*dxi[2],                 F(2,2)*dxi[2],
                        F(0,0)*dxi[1] + F(0,1)*dxi[0], F(1,0)*dxi[1] + F(1,1)*dxi[0], F(2,0)*dxi[1] + F(2,1)*dxi[0],
                        F(0,1)*dxi[2] + F(0,2)*dxi[1], F(1,1)*dxi[2] + F(1,2)*dxi[1], F(2,1)*dxi[2] + F(2,2)*dxi[1],
                        F(0,0)*dxi[2] + F(0,2)*dxi[0], F(1,0)*dxi[2] + F(1,2)*dxi[0], F(2,0)*dxi[2] + F(2,2)*dxi[0];

                // The 3x3 sub-matrix Kii is symmetric, we only store its upper triangular part
                Mat33 Kii = (dxi.dot(S*dxi)*Id + Bi.transpose()*D*Bi) * detJ * w;
                Ke.template block<Dimension, Dimension>(i*Dimension, i*Dimension)
                        .template triangularView<Eigen::Upper>()
                        += Kii;

                // We now loop only on the upper triangular part of the
                // element stiffness matrix Ke since it is symmetric
                for (std::size_t j = i+1; j < NumberOfNodesPerElement; ++j) {
                    // Derivatives of the jth shape function at the gauss node with respect to global coordinates x,y and z
                    const Vector<3> dxj = dN_dx.row(j).transpose();

                    Matrix<6,3> Bj;
                    Bj <<
                       F(0,0)*dxj[0],                 F(1,0)*dxj[0],                 F(2,0)*dxj[0],
                            F(0,1)*dxj[1],                 F(1,1)*dxj[1],                 F(2,1)*dxj[1],
                            F(0,2)*dxj[2],                 F(1,2)*dxj[2],                 F(2,2)*dxj[2],
                            F(0,0)*dxj[1] + F(0,1)*dxj[0], F(1,0)*dxj[1] + F(1,1)*dxj[0], F(2,0)*dxj[1] + F(2,1)*dxj[0],
                            F(0,1)*dxj[2] + F(0,2)*dxj[1], F(1,1)*dxj[2] + F(1,2)*dxj[1], F(2,1)*dxj[2] + F(2,2)*dxj[1],
                            F(0,0)*dxj[2] + F(0,2)*dxj[0], F(1,0)*dxj[2] + F(1,2)*dxj[0], F(2,0)*dxj[2] + F(2,2)*dxj[0];

                    // The 3x3 sub-matrix Kij is NOT symmetric, we store its full part
                    Mat33 Kij = (dxi.dot(S*dxj)*Id + Bi.transpose()*D*Bj) * detJ * w;
                    Ke.template block<Dimension, Dimension>(i*Dimension, j*Dimension)
                            .noalias() += Kij;
                }
            }
        }

#pragma omp critical
        for (std::size_t i = 0; i < NumberOfNodesPerElement; ++i) {
            // Node index of the ith node in the global stiffness matrix
            const auto x = static_cast<int>(node_indices[i]*Dimension);
            for (int m = 0; m < Dimension; ++m) {
                for (int n = m; n < Dimension; ++n) {
                    triplets.emplace_back(x+m, x+n, Ke(i*Dimension+m,i*Dimension+n));
                }
            }

            for (std::size_t j = i+1; j < NumberOfNodesPerElement; ++j) {
                // Node index of the jth node in the global stiffness matrix
                const auto y = static_cast<int>(node_indices[j]*Dimension);
                for (int m = 0; m < Dimension; ++m) {
                    for (int n = 0; n < Dimension; ++n) {
                        triplets.emplace_back(x+m, y+n, Ke(i*Dimension+m,j*Dimension+n));
                    }
                }
            }
        }
    }
    p_K.setFromTriplets(triplets.begin(), triplets.end());
}


void SVKElasticForcefield_SOFA::addDForce(const MechanicalParams* /*mparams*/,
                                          Data<VecDeriv>& /*d_df*/,
                                          const Data<VecDeriv>& /*d_dx*/) {
    // Here you would compute df = K*dx
}

void SVKElasticForcefield_SOFA::draw(const sofa::core::visual::VisualParams *vparams) {

    if (!this->mstate.get() || !d_topology_container.get()) {
        return;
    }

    if (!vparams->displayFlags().getShowForceFields()) {
        return;
    }

    auto * state = this->mstate.get();

    vparams->drawTool()->disableLighting();
    const auto x = state->readPositions();

    std::vector< sofa::type::Vec<3, double> > points[4];
    const auto number_of_elements = this->topology()->number_of_elements();
    for (sofa::core::topology::Topology::TetrahedronID i = 0 ; i<number_of_elements;++i) {
        const auto t=this->topology()->domain()->element_indices(i);

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

void SVKElasticForcefield_SOFA::computeBBox(const sofa::core::ExecParams * /*params*/, bool onlyVisible) {
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
 .add<SVKElasticForcefield_SOFA>();
