import React, { Suspense } from "react";
import { RouteObject } from "react-router-dom";
import { Spin } from "antd";
import AppShell from "@/components/layout/AppShell";
import ProtectedRoute from "./ProtectedRoute";
import RoleGuard from "./RoleGuard";
import { UserRole } from "@/types/enums";

// Lazy-loaded pages
const LoginPage = React.lazy(() => import("@/pages/auth/LoginPage"));
const RegisterPage = React.lazy(() => import("@/pages/auth/RegisterPage"));
const DashboardPage = React.lazy(
  () => import("@/pages/dashboard/DashboardPage")
);
const DetectionPage = React.lazy(
  () => import("@/pages/detection/DetectionPage")
);
const TaxonomyPage = React.lazy(
  () => import("@/pages/taxonomy/TaxonomyPage")
);
const RegexRulesPage = React.lazy(
  () => import("@/pages/rules/RegexRulesPage")
);
const FieldPatternsPage = React.lazy(
  () => import("@/pages/rules/FieldPatternsPage")
);
const ContextRulesPage = React.lazy(
  () => import("@/pages/rules/ContextRulesPage")
);
const MaskingRulesPage = React.lazy(
  () => import("@/pages/rules/MaskingRulesPage")
);
const PendingRulesPage = React.lazy(
  () => import("@/pages/learning/PendingRulesPage")
);
const TrainingDataPage = React.lazy(
  () => import("@/pages/learning/TrainingDataPage")
);
const ModelsPage = React.lazy(() => import("@/pages/models/ModelsPage"));
const APIKeysPage = React.lazy(
  () => import("@/pages/api-management/APIKeysPage")
);
const AccessRequestsPage = React.lazy(
  () => import("@/pages/api-management/AccessRequestsPage")
);
const APIUsagePage = React.lazy(
  () => import("@/pages/api-management/APIUsagePage")
);
const UsersPage = React.lazy(() => import("@/pages/users/UsersPage"));
const AuditPage = React.lazy(() => import("@/pages/audit/AuditPage"));
const SettingsPage = React.lazy(
  () => import("@/pages/settings/SettingsPage")
);
const ProfilePage = React.lazy(
  () => import("@/pages/profile/ProfilePage")
);

const SuspenseWrapper: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => (
  <Suspense
    fallback={
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          height: "100%",
          padding: "48px",
        }}
      >
        <Spin size="large" />
      </div>
    }
  >
    {children}
  </Suspense>
);

export const routes: RouteObject[] = [
  // Public routes
  {
    path: "/login",
    element: (
      <SuspenseWrapper>
        <LoginPage />
      </SuspenseWrapper>
    ),
  },
  {
    path: "/register",
    element: (
      <SuspenseWrapper>
        <RegisterPage />
      </SuspenseWrapper>
    ),
  },

  // Protected routes
  {
    element: (
      <ProtectedRoute>
        <AppShell />
      </ProtectedRoute>
    ),
    children: [
      {
        path: "/dashboard",
        element: (
          <SuspenseWrapper>
            <DashboardPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: "/detection",
        element: (
          <SuspenseWrapper>
            <DetectionPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: "/taxonomy",
        element: (
          <SuspenseWrapper>
            <TaxonomyPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: "/rules/regex",
        element: (
          <SuspenseWrapper>
            <RegexRulesPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: "/rules/field-patterns",
        element: (
          <SuspenseWrapper>
            <FieldPatternsPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: "/rules/context",
        element: (
          <SuspenseWrapper>
            <ContextRulesPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: "/rules/masking",
        element: (
          <SuspenseWrapper>
            <MaskingRulesPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: "/learning/pending-rules",
        element: (
          <SuspenseWrapper>
            <PendingRulesPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: "/learning/training-data",
        element: (
          <SuspenseWrapper>
            <TrainingDataPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: "/models",
        element: (
          <SuspenseWrapper>
            <ModelsPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: "/api/keys",
        element: (
          <SuspenseWrapper>
            <APIKeysPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: "/api/requests",
        element: (
          <SuspenseWrapper>
            <AccessRequestsPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: "/api/usage",
        element: (
          <SuspenseWrapper>
            <APIUsagePage />
          </SuspenseWrapper>
        ),
      },
      {
        path: "/users",
        element: (
          <SuspenseWrapper>
            <RoleGuard minRole={UserRole.TENANT_ADMIN}>
              <UsersPage />
            </RoleGuard>
          </SuspenseWrapper>
        ),
      },
      {
        path: "/audit",
        element: (
          <SuspenseWrapper>
            <RoleGuard minRole={UserRole.TENANT_ADMIN}>
              <AuditPage />
            </RoleGuard>
          </SuspenseWrapper>
        ),
      },
      {
        path: "/settings",
        element: (
          <SuspenseWrapper>
            <SettingsPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: "/settings/:tab",
        element: (
          <SuspenseWrapper>
            <SettingsPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: "/profile",
        element: (
          <SuspenseWrapper>
            <ProfilePage />
          </SuspenseWrapper>
        ),
      },
    ],
  },
];
